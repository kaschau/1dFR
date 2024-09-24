import numpy as np
from oneDFR.poly import LegendrePoly
from oneDFR.integrators import BaseIntegrator
from oneDFR.flux import BaseFlux
from oneDFR.util import subclass_where
from pathlib import Path
from oneDFR.output import plot

np.seterr(all="raise")
fpdtype_max = np.finfo(np.float64).max
fpdtype_min = np.finfo(np.float64).eps


def noop(*args, **kwargs):
    pass


def get_quad_rules(config):
    p = config["p"]
    rule = config["quad"]
    if "lobatto" in rule:
        fname = f"/quadrules/gauss-legendre-lobatto-n{p+1}-d{2*(p)-1}-spu.txt"
        direct = str(Path(__file__).parent)
        with open(direct + fname) as f:
            data = np.genfromtxt(f, delimiter=" ")
            return data[:, 0]
    else:
        data = np.polynomial.legendre.leggauss(p + 1)
        return data[0]


def vcjg(k, c, x, der=False):

    if c == 0:
        etak = 0.0
    elif c == 1:
        etak = k / (k + 1)
    elif c == 2:
        etak = (k + 1) / k
    else:
        raise ValueError

    Legk = LegendrePoly(k)
    Legkm = LegendrePoly(k - 1)
    Legkp = LegendrePoly(k + 1)

    def g(x, etak):
        return 0.5 * (Lk(x) + (etak * Lkm(x) + Lkp(x)) / (1 + etak))

    if not der:
        Lk = Legk.basis_at
        Lkm = Legkm.basis_at
        Lkp = Legkp.basis_at
        gr = g(x, etak)
        gl = g(-x, etak)
    else:
        Lk = Legk.dbasis_at
        Lkm = Legkm.dbasis_at
        Lkp = Legkp.dbasis_at
        gr = g(x, etak)
        gl = -g(-x, etak)

    return gl, gr


class system:
    def __init__(self, config):
        self.config = config
        self.t = 0.0
        self.niter = 0

        self.nvar = nvar = 3
        self.order = order = config["p"]
        self.nupts = nupts = order + 1

        # solution and flux deriv polys
        self.upoly = LegendrePoly(order)
        self.dfpoly = LegendrePoly(order - 1)

        # get num solution points
        self.upts = get_quad_rules(config)

        # test if flux points are part of solution points
        if -1 in self.upts:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        # create solution point inverse/vandermonde matrix
        self.uvdm = self.upoly.vandermonde(self.upts)
        self.invuvdm = np.linalg.inv(self.uvdm)

        # left and right solution vandermonde
        self.lvdm = self.upoly.vandermonde([-1])
        self.rvdm = self.upoly.vandermonde([1])

        # flux derivative vandermonde
        self.dfvdm = self.dfpoly.vandermonde(self.upts)

        # compute g' of correction functions at solution points
        c = 0  # Vincent constant 0 = nodal DG
        self.gL, self.gR = vcjg(order, c, self.upts, der=True)

        # set interpolation to face
        if self.fpts_in_upts:
            self.u_to_f = self._u_to_f_closed
        else:
            self.u_to_f = self._u_to_f_open

        # SET BOUNDARY CONDITIONS
        self.bc = getattr(self, f"_bc_{config["bc"]}")

        # set flux
        self.flux = subclass_where(BaseFlux, name=config["intflux"])(config)

        # read grid
        self.read_grid()
        neles = self.neles

        # allocate arrays
        # solution modes
        self.ua = np.zeros((nvar, nupts, neles))
        # flux poly'l modes
        self.fa = np.zeros((nvar, nupts, neles))
        # solution @ Xi=1 (left side of interfaces)
        self.uL = np.zeros((nvar, 1, neles + 1))
        # solution @ Xi=-1 (right side of interfaces)
        self.uR = np.zeros((nvar, 1, neles + 1))
        # continuous flux values
        self.fc = np.zeros((nvar, 1, neles + 1))
        # flux derivative modes
        self.dfa = np.zeros((nvar, nupts - 1, neles))
        # flux polyl @ Xi=-1
        self.fl = np.zeros((nvar, 1, neles))
        # flux polyl @ Xi=1
        self.fr = np.zeros((nvar, 1, neles))

        # create integrator
        self.intg = subclass_where(BaseIntegrator, name=config["intg"])()
        for bank in range(self.intg.nbanks):
            setattr(self, f"u{bank}", np.zeros((nvar, nupts, neles)))

        # see if we are filtering
        self.efilt = self.config["efilt"]
        if self.efilt is not None and self.order > 0:

            # create the ef_pts
            if self.fpts_in_upts:
                self.efpts = self.upts
                self.nefpts = len(self.efpts)
                self.efvdm = self.uvdm
                self.invefvdm = self.invuvdm
            else:
                self.efpts = np.concatenate((self.upts, [-1, 1]))
                self.nefpts = len(self.efpts)
                fdm = self.upoly.vandermonde([-1, 1])
                self.efvdm = np.vstack((self.uvdm, fdm))

            self.entmin_int = np.zeros((2, neles))
            self.entropy = getattr(self, f"_entropy_{config["effunc"]}")
            self.intcent = self._intcent
            self.entropy_local = self._entropy_local
            if self.efilt == "bisect":
                self.entropy_filter = self._entropy_filter_bisect
                self.get_minima = self._get_minima_bisect
            elif self.efilt == "linearise":
                self.entropy_filter = self._entropy_filter_linearise
                self.get_minima = self._get_minima_linearise
            else:
                raise ValueError("What entropy filter?")
            self.bcent = self._bcent
        else:
            self.entropy = noop
            self.intcent = noop
            self.entropy_local = noop
            self.entropy_filter = noop
            self.bcent = noop

    def _entropy_numerical(self, u):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape) * fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = np.log(p[idx]) - gamma * np.log(rho[idx])

        return e

    def _entropy_physical(self, u):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape) * fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = p / rho**gamma

        return e

    def _entropy_numerical_dim(self, u):
        # cp = 1000.0
        cv = 714.2857142857143
        # R = cp - cv
        R = 285.71428571428567
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape) * fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = cv * (np.log(p[idx]) - gamma * np.log(rho[idx]) - np.log(R))

        return e

    def _entropy_local(self, ubank):
        # assume ubank are current
        u = getattr(self, f"u{ubank}")

        # compute element entropy
        self.entmin_int[:] = np.min(self.entropy(u), axis=0)[np.newaxis, :]

        # compute interface entropy
        if not self.fpts_in_upts:
            self.u_to_f()
            self.entmin_int[:] = np.minimum(
                np.min(self.entropy(self.uL[:, :, 1::]), axis=0), self.entmin_int
            )
            self.entmin_int[:] = np.minimum(
                np.min(self.entropy(self.uR[:, :, 0:-1]), axis=0), self.entmin_int
            )

    def _intcent(self):
        self.entmin_int[0, 1:-1] = np.minimum(
            self.entmin_int[0, 1:-1], self.entmin_int[1, 0:-2]
        )
        self.entmin_int[1, 0:-2] = np.minimum(
            self.entmin_int[1, 0:-2], self.entmin_int[0, 1:-1]
        )

    def _get_minima_bisect(self, u, modes):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = self.entropy(u)

        rho = np.min(rho, axis=0)
        p = np.min(p, axis=0)
        e = np.min(e, axis=0)

        if not self.fpts_in_upts:
            # interpolate to faces
            uL = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uL, self.rvdm, modes)
            uR = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uR, self.lvdm, modes)

            rhoL = uL[0]
            vL = uL[1] / rhoL
            rhoEL = uL[2]
            pL = (gamma - 1.0) * (rhoEL - 0.5 * rhoL * vL**2)
            eL = self.entropy(uL)

            rho = np.minimum(rho, np.min(rhoL, axis=0))
            p = np.minimum(p, np.min(pL, axis=0))
            e = np.minimum(e, np.min(eL, axis=0))

            rhoR = uR[0]
            vR = uR[1] / rhoR
            rhoER = uR[2]
            pR = (gamma - 1.0) * (rhoER - 0.5 * rhoR * vR**2)
            eR = self.entropy(uR)

            rho = np.minimum(rho, np.min(rhoR, axis=0))
            p = np.minimum(p, np.min(pR, axis=0))
            e = np.minimum(e, np.min(eR, axis=0))

        return rho, p, e

    def _get_minima_linearise(self, u, modes):
        rho = u[0]
        v = u[1] / rho
        rhoe = u[2] - 0.5 * rho * v**2

        e = self.entropy(u)

        rho = np.min(rho, axis=0)
        rhoe = np.min(rhoe, axis=0)
        e = np.min(e, axis=0)

        if not self.fpts_in_upts:
            # interpolate to faces
            uL = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uL, self.rvdm, modes)
            uR = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uR, self.lvdm, modes)

            rhoL = uL[0]
            vL = uL[1] / rhoL
            rhoeL = uL[2] - 0.5 * rhoL * vL**2
            eL = self.entropy(uL)

            rho = np.minimum(rho, np.min(rhoL, axis=0))
            rhoe = np.minimum(rhoe, np.min(rhoeL, axis=0))
            e = np.minimum(e, np.min(eL, axis=0))

            rhoR = uR[0]
            vR = uR[1] / rhoR
            rhoeR = uR[2] - 0.5 * rhoR * vR**2
            eR = self.entropy(uR)

            rho = np.minimum(rho, np.min(rhoR, axis=0))
            rhoe = np.minimum(rhoe, np.min(rhoeR, axis=0))
            e = np.minimum(e, np.min(eR, axis=0))

        return rho, rhoe, e

    def filter_single(self, umt, ui, f, uidx):
        pmax = self.order + 1
        vrho = v2rho = vmom = v2mom = vE = v2E = 1.0
        for p in range(1, pmax):
            v2rho *= vrho * vrho * f ** self.config["efrhopow"]
            v2mom *= vmom * vmom * f ** self.config["efmompow"]
            v2E *= vE * vE * f ** self.config["efEpow"]
            vrho *= f ** self.config["efrhopow"]
            vmom *= f ** self.config["efmompow"]
            vE *= f ** self.config["efEpow"]
            umt[0, p] *= v2rho
            umt[1, p] *= v2mom
            umt[2, p] *= v2E
        # get new solution
        self.upoly.evaluate(ui, self.efvdm[uidx : uidx + 1], umt)

        rho = ui[0]
        v = ui[1] / rho
        rhoE = ui[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = self.entropy(ui)

        return rho[0], p[0], e[0]

    def filter_full(self, umt, unew, f):
        pmax = self.order + 1
        vrho = v2rho = vmom = v2mom = vE = v2E = 1.0
        for p in range(1, pmax):
            v2rho *= vrho * vrho * f ** self.config["efrhopow"]
            v2mom *= vmom * vmom * f ** self.config["efmompow"]
            v2E *= vE * vE * f ** self.config["efEpow"]
            vrho *= f ** self.config["efrhopow"]
            vmom *= f ** self.config["efmompow"]
            vE *= f ** self.config["efEpow"]
            umt[0, p] *= v2rho
            umt[1, p] *= v2mom
            umt[2, p] *= v2E
        # get new solution
        self.upoly.evaluate(unew, self.uvdm, umt)

        return self.get_minima(unew, umt)

    def _entropy_filter_bisect(self, ubank):
        # assumes entmin_int is already populated
        u = getattr(self, f"u{ubank}")

        try:
            d_min = self.config["d_min"]
        except KeyError:
            d_min = 1e-6
        try:
            p_min = self.config["p_min"]
        except KeyError:
            p_min = 1e-6
        try:
            e_tol = self.config["e_tol"]
        except KeyError:
            e_tol = 0.0
        try:
            f_tol = self.config["f_tol"]
        except KeyError:
            f_tol = 1e-4

        # get min entropy for element and neighbors
        entmin = np.min(self.entmin_int, axis=0)

        # compute rho, p, e for all elements
        dmin, pmin, emin = self.get_minima(u, self.ua)

        filtidx = np.where(
            np.bitwise_or(
                dmin < d_min, np.bitwise_or(pmin < p_min, emin < entmin - e_tol)
            )
        )[0]

        for idx in filtidx:
            umodes = self.ua[:, :, idx : idx + 1]
            unew = np.copy(u[:, :, idx : idx + 1])

            f = 1.0

            for uidx in range(self.nefpts):
                if uidx < self.nupts:
                    ui = np.copy(unew[:, uidx : uidx + 1])
                else:
                    ui = np.zeros((self.nvar, 1, 1))
                    self.upoly.evaluate(ui, self.efvdm[uidx : uidx + 1], umodes)

                # Do da filter with current f for this solution point
                d, p, e = self.filter_single(np.copy(umodes), ui, f, uidx)

                if d < d_min or p < p_min or e < entmin[idx] - e_tol:

                    # Setup root finding interval
                    flow = 0.0
                    fhigh = f

                    # Iterate on filter strength
                    for i in range(self.config["efniter"]):

                        # define new f
                        f = 0.5 * (flow + fhigh)

                        d, p, e = self.filter_single(np.copy(umodes), unew, f, uidx)

                        if d < d_min or p < p_min or e < entmin[idx] - e_tol:
                            fhigh = f
                        else:
                            flow = f

                        if fhigh - flow < f_tol:
                            break

                    f = flow

            umodes = self.ua[:, :, idx : idx + 1]
            ## Filter entire solution with flow
            dmin, pmin, e_min = self.filter_full(np.copy(umodes), unew, f)
            emin[idx] = e_min[0]

            # Update final solution with filtered values
            u[:, :, idx : idx + 1] = unew

            # update modes
            self.upoly.compute_coeff(self.ua[:, :, idx : idx + 1], unew, self.invuvdm)

        # update min interface entropy
        self.entmin_int[:] = emin[np.newaxis, :]

    def _entropy_filter_linearise(self, ubank):
        # assumes entmin_int is already populated
        u = getattr(self, f"u{ubank}")

        try:
            d_min = self.config["d_min"]
        except KeyError:
            d_min = 1e-6
        try:
            rhoe_min = self.config["rhoe_min"]
        except KeyError:
            rhoe_min = 1e-6
        try:
            e_tol = self.config["e_tol"]
        except KeyError:
            e_tol = 0.0

        # get min entropy for element and neighbors
        entmin = np.min(self.entmin_int, axis=0)

        # compute rho, p, e for all elements
        dmin, rhoemin, emin = self.get_minima(u, self.ua)

        filtidx = np.where(
            np.bitwise_or(
                dmin < d_min,
                np.bitwise_or(
                    rhoemin < rhoe_min,
                    np.min(u[0, :, :] * (self.entropy(u) - entmin), axis=0) < e_tol,
                ),
            )
        )[0]

        for idx in filtidx:
            umodes = self.ua[:, :, idx : idx + 1]

            if self.fpts_in_upts:
                ui = np.copy(u[:, :, idx : idx + 1])
            else:
                ui = np.zeros((self.nvar, self.nefpts, 1))
                self.upoly.evaluate(ui, self.efvdm, umodes)

            # First test for negative density
            dmin = np.min(ui[0, :, 0])
            if dmin < d_min:
                theta = (umodes[0, 0, 0] - d_min) / (umodes[0, 0, 0] - dmin)
                theta = min(1.0, max(theta, 0.0))
                ui[0, :, 0] = umodes[0, 0, 0] + theta * (ui[0, :, 0] - umodes[0, 0, 0])
                self.upoly.compute_coeff(umodes, ui[:, 0 : self.nupts], self.invuvdm)
                assert np.min(ui[0, :, 0]) >= d_min

            # Now test for negative internal energy
            rhoemin = np.min(ui[2, :, 0] - 0.5 * ui[1, :, 0] ** 2 / ui[0, :, 0])
            if rhoemin < rhoe_min:
                rhoeave = umodes[2, 0, 0] - 0.5 * umodes[1, 0, 0] ** 2 / umodes[0, 0, 0]
                theta = (rhoeave - rhoe_min) / (rhoeave - rhoemin)
                theta = np.power(
                    np.ones(self.nvar) * min(1.0, max(theta, 0.0)),
                    [
                        self.config["efrhopow"],
                        self.config["efmompow"],
                        self.config["efEpow"],
                    ],
                )
                ui[:, :, 0] = umodes[:, 0, 0][:, np.newaxis] + theta[:, np.newaxis] * (
                    ui[:, :, 0] - umodes[:, 0, 0][:, np.newaxis]
                )
                self.upoly.compute_coeff(umodes, ui[:, 0 : self.nupts], self.invuvdm)
                assert (
                    np.min(ui[2, :, 0] - 0.5 * ui[1, :, 0] ** 2 / ui[0, :, 0])
                    >= rhoe_min
                )

            # Finally, test for entropy
            ei = self.entropy(ui[:, :, 0])
            Xmin = min(ui[0, :, 0] * (ei - entmin[idx]))
            if Xmin < e_tol:
                Xavg = umodes[0, 0, 0] * (self.entropy(umodes[:, 0, :]) - entmin[idx])
                theta = Xavg / max(
                    Xavg - np.min(ui[0, :, 0] * (ei - entmin[idx])), 1e-16
                )
                theta = np.power(
                    np.ones(self.nvar) * min(1.0, max(theta, 0.0)),
                    [
                        self.config["efrhopow"],
                        self.config["efmompow"],
                        self.config["efEpow"],
                    ],
                )
                ui[:, :, 0] = umodes[:, 0, 0][:, np.newaxis] + theta[:, np.newaxis] * (
                    ui[:, :, 0] - umodes[:, 0, 0][:, np.newaxis]
                )
                self.upoly.compute_coeff(umodes, ui[:, 0 : self.nupts], self.invuvdm)
                ei = self.entropy(ui[:, :, 0])
                # test = np.min(ui[0, :, 0] * (ei - entmin[idx]))
                # assert test >= 0.0

            # Update solution
            u[:, :, idx : idx + 1] = ui[:, 0 : self.nupts, :]

            # update modes
            self.ua[:, :, idx : idx + 1] = umodes

            # update min interface entropy
            emin = min(self.entropy(ui[:, :, 0]))
            self.entmin_int[:, idx : idx + 1] = emin

    def _u_to_f_closed(self, ubank):
        u = getattr(self, f"u{ubank}")
        # REMEMBER, RIGHT interface is LEFT side of element
        self.uL[:, 0, 1::] = u[:, -1, :]
        self.uR[:, 0, 0:-1] = u[:, 0, :]

    def _u_to_f_open(self, *args):
        # REMEMBER, RIGHT interface is LEFT side of element
        # interpolate solution to left face
        self.upoly.evaluate(self.uL[:, :, 1::], self.rvdm, self.ua)
        # interpolate solution to right face
        self.upoly.evaluate(self.uR[:, :, 0:-1], self.lvdm, self.ua)

    def _bc_wall(self):
        self.uL[:, :, 0] = self.uR[:, :, 0]
        self.uR[:, :, -1] = self.uL[:, :, -1]

    def _bc_periodic(self):
        self.uL[:, :, 0] = self.uL[:, :, -1]
        self.uR[:, :, -1] = self.uR[:, :, 0]

    def _bcent(self):
        uL = self.uL[:, :, 0]
        eL = self.entropy(uL)
        self.entmin_int[0, 0] = min(eL[0], self.entmin_int[0, 0])

        uR = self.uR[:, :, -1]
        eR = self.entropy(uR)
        self.entmin_int[-1, -1] = min(eR[0], self.entmin_int[-1, -1])

    def update_solution_stuff(self, ubank):
        u = getattr(self, f"u{ubank}")
        # create solution poly'l
        self.upoly.compute_coeff(self.ua, u, self.invuvdm)

        self.entropy_filter(ubank)

        # interpolate solution to face
        self.u_to_f(ubank)

        self.intcent()
        # compute bcs
        self.bc()
        self.bcent()

    def update_flux_stuff(self, ubank, fbankout):
        u = getattr(self, f"u{ubank}")
        f = getattr(self, f"u{fbankout}")
        # compute pointwise fluxes
        self.flux.flux(u, f)

        # compute flux poly'l coeffs
        self.upoly.compute_coeff(self.fa, f, self.invuvdm)

        # compute common fluxes
        self.flux.intflux(self.uL, self.uR, self.fc)

    def build_negdivconf(self, fbankout):
        # Begin building of negdivconf
        negdivconf = getattr(self, f"u{fbankout}")

        # compute flux derivative at solution points
        self.dfa[:] = self.dfpoly.diff_coeff(self.fa)
        self.dfpoly.evaluate(negdivconf, self.dfvdm, self.dfa)

        # evaluate + add the left jumps to negdivconf
        self.upoly.evaluate(self.fl, self.lvdm, self.fa)
        negdivconf += np.einsum(
            "vx...,x...->vx...", self.fc[:, :, 0:-1] - self.fl, self.gL
        )

        # eval + add the right jumps to negdivconf
        self.upoly.evaluate(self.fr, self.rvdm, self.fa)
        negdivconf += np.einsum(
            "vx...,x...->vx...", self.fc[:, :, 1::] - self.fr, self.gR
        )

        # transform to neg flux in physical coords
        negdivconf *= -self.invJac

    def RHS(self, ubankin, fbankout):

        # assumes ubank is up to date
        self.update_solution_stuff(ubankin)

        self.update_flux_stuff(ubankin, fbankout)

        self.build_negdivconf(fbankout)

    def read_grid(self):
        fname = self.config["mesh"]
        with open(fname, "rb") as f:
            eles = np.load(f)
        h = eles[:, 1] - eles[:, 0]
        self.x = np.mean(eles, axis=-1)[np.newaxis, :] + np.einsum(
            "i,j->ij", self.upts, h / 2.0
        )
        self.invJac = 2.0 / h
        self.neles = np.shape(eles)[0]

    def set_ics(self, pris):
        ubank = 0
        u = getattr(self, f"u{ubank}")
        # density
        u[0, :] = pris[0]
        # momentum
        u[1, :] = pris[1] * pris[0]
        # total energy
        u[2, :] = 0.5 * pris[0] * pris[1] ** 2 + pris[2] / (self.config["gamma"] - 1.0)

        # prepare for first iteration
        self.upoly.compute_coeff(self.ua, u, self.invuvdm)
        self.u_to_f(0)
        self.bc()

        self.entropy_local(0)
        self.bcent()
        self.entropy_filter(0)

    def run(self):
        while round(self.t, 5) <= self.config["tend"]:
            try:
                if self.niter % self.config["nout"] == 0:
                    plot(self, f"{self.config["outfname"]}_{self.niter:06d}.png")
            except ZeroDivisionError:
                pass
            self.intg.step(self, self.config["dt"])


if __name__ == "__main__":
    config = {
        "p": 3,
        "quad": "gauss-legendre",
        "intg": "rk3",
        "intflux": "hllc",
        "gamma": 1.4,
        "nout": 0,
        "bc": "wall",
        "mesh": "mesh-50.npy",
        "dt": 1e-4,
        "tend": 0.2,
        "outfname": "oneD",
        "efilt": "linearise",
        "effunc": "numerical_dim",
        "efniter": 20,
        "efrhopow": 1.0,
        "efmompow": 1.0,
        "efEpow": 1.0,
    }

    neles = int(config["mesh"].split("-")[1].split(".")[0])
    half = int(neles / 2)

    rho = np.zeros((config["p"] + 1, neles))
    rho[:, 0:half] = 1.0
    rho[:, half::] = 0.125

    v = 0

    p = np.zeros(rho.shape)
    p[:, 0:half] = 1.0
    p[:, half::] = 0.1

    a = system(config)
    a.set_ics([rho, v, p])
    a.run()
    plot(a)

import numpy as np
from oneDFR.poly import LegendrePoly
from oneDFR.integrators import BaseIntegrator
from oneDFR.flux import BaseFlux
from oneDFR.util import subclass_where
from pathlib import Path
from oneDFR.output import plot

np.seterr(all="raise")
fpdtype_max = 1e10  # np.finfo(np.float64).max
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

        self.nvar = nvar = 3  # 1D
        self.nfpts = nfpts = 2  # 1D
        self.order = order = config["p"]
        self.nupts = nupts = order + 1

        # solution space polynomial
        self.ppoly = LegendrePoly(order)
        # derivative space polynomial
        self.dppoly = LegendrePoly(order - 1)

        # get solution points in transformed space
        self.upts = get_quad_rules(config)

        # read grid
        self.read_grid()
        neles = self.neles

        # create solution point inverse/vandermonde matrix
        # Generalized vandermond V = \Psi(~x)
        # Evaluated at solution points
        # and its inverse, V^{-1}
        # [nupts x p+1] = [nupts x nupts]
        self.uvdm = self.ppoly.vandermonde(self.upts)
        self.invuvdm = np.linalg.inv(self.uvdm)

        # **********************************************************
        # CONSTRUCT NODAL BASIS SET OPERATOR M0 = lu(~xf) = V^{-1}V(~xf)
        # An operator to interpolate the solution to the flux points
        # i.e. M0 * u = u_f
        # M0 => [nfpts x nupts]
        # u => [nvar x nupts]
        # u_f => [nvar x nfpts]
        # **********************************************************

        # Now we need V(~xf)
        lvdm = self.ppoly.vandermonde(-1)[0]
        rvdm = self.ppoly.vandermonde(1)[0]
        Vxf = np.array([lvdm, rvdm])

        self.M0 = Vxf @ self.invuvdm

        # **********************************************************
        # CONSTRUCT GRADIENT of NODAL BASIS SET OPERATOR
        # M1 = grad lu(~xu) = V^{-1}V'(~xu)
        # An operator to compute the derivative of the function at solution
        # points
        # i.e. M1 * f = f'
        # M1 => [nupts x nupts]
        # f => [nvar x nupts]
        # f' => [nvar x nupts]
        # **********************************************************
        def dVdx(x):
            p = len(x) - 1
            dV = np.zeros((p + 1, p + 1))
            if p > 0:
                dV[:, 1] = 1.0
            for row in range(p + 1):
                for column in range(2, p + 1):
                    dV[row, column] = column * x[row] ** (column - 1)

            return dV

        dV = dVdx(self.upts)
        self.M1 = dV @ self.invuvdm

        # in 1D, M2 = M0 since the normal is just one.
        self.M2 = self.M0

        # **********************************************************
        # CONSTRUCT GRADIENT OF CORRECTION FUNCTION AT SOLUTION POINTS
        # compute g' of correction functions at solution points in transformed space
        c = 0  # Vincent constant 0 = nodal DG
        self.gL, self.gR = vcjg(order, c, self.upts, der=True)
        # [nfpts x nupts]
        self.M3 = np.array([self.gL, self.gR])

        # solution derivative space vandermonde at flux points
        # [nfpts x nupts - 1]
        # NEW FOR NSCBC
        self.Minf = np.array(
            [self.dppoly.vandermonde(-1)[0], self.dppoly.vandermonde(1)[0]]
        )

        # compute derivative of correction functions at flux points in transformed space
        self.gLf, _ = vcjg(order, c, np.array([-1]), der=True)
        _, self.gRf = vcjg(order, c, np.array([1]), der=True)
        # [nfpts x nupts]
        self.M6 = np.array([self.gLf, self.gRf])

        # create integrator and state arrays
        # [nvar x nupts x neles]
        self.intg = subclass_where(BaseIntegrator, name=config["intg"])()
        for bank in range(self.intg.nbanks):
            setattr(self, f"u{bank}", np.zeros((nvar, nupts, neles)))

        # test if flux points are subset of solution points
        if -1 in self.upts:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        # set interpolation to face
        if self.fpts_in_upts:
            self.u_to_f = self._u_to_f_closed
        else:
            self.u_to_f = self._u_to_f_open

        # SET BOUNDARY CONDITIONS
        self.bcl = getattr(self, f"_bc_{config["bcl"]}")
        self.bcr = getattr(self, f"_bc_{config["bcr"]}")

        # set flux
        self.flux = subclass_where(BaseFlux, name=config["intflux"])(config)

        # allocate arrays
        # solution modes
        self.ua = np.zeros((nvar, nupts, neles))

        # solution flux points
        self.uf = np.zeros((nvar, nfpts, neles))

        # continuous flux values
        self.fc = np.zeros((nvar, nfpts, neles))

        # flux polyl @ Xi=-1
        # flux polyl @ Xi=1
        self.ff = np.zeros((nvar, nfpts, neles))

        # solution derivative on left and right face
        self.duL = np.zeros(nvar)
        self.duR = np.zeros(nvar)

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
            self.bcentl = getattr(self, f"_bc_ent_{config["bcl"]}")
            self.bcentr = getattr(self, f"_bc_ent_{config["bcr"]}")
        else:
            self.entropy = noop
            self.intcent = noop
            self.entropy_local = noop
            self.entropy_filter = noop
            self.bcentl = noop
            self.bcentr = noop

    def _entropy_nondim(self, u):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape) * fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = np.log(p[idx]) - gamma * np.log(rho[idx])

        return e

    def _entropy_nondim_exp(self, u):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape) * fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = p[idx] * rho[idx] ** -gamma

        return e

    def _entropy_dim(self, u):
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
        # assume ubank is current
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
        self.entmin_int[0, 1::] = np.minimum(
            self.entmin_int[0, 1::], self.entmin_int[1, 0:-1]
        )
        self.entmin_int[1, 0:-1] = np.minimum(
            self.entmin_int[1, 0:-1], self.entmin_int[0, 1::]
        )

    def _get_minima_bisect(self, u, modes, entmin):
        nupts = self.nupts
        rho = u[0, 0:nupts]
        v = u[1, 0:nupts] / rho
        rhoE = u[2, 0:nupts]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = self.entropy(u[:, 0:nupts])

        rho = np.min(rho, axis=0)
        p = np.min(p, axis=0)

        X = np.min(self._chi(u[:, 0:nupts], e, entmin), axis=0)
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
            X = np.minimum(X, np.min(self._chi(uL, eL, entmin), axis=0))
            e = np.minimum(e, np.min(eL, axis=0))

            rhoR = uR[0]
            vR = uR[1] / rhoR
            rhoER = uR[2]
            pR = (gamma - 1.0) * (rhoER - 0.5 * rhoR * vR**2)
            eR = self.entropy(uR)

            rho = np.minimum(rho, np.min(rhoR, axis=0))
            p = np.minimum(p, np.min(pR, axis=0))
            X = np.minimum(X, np.min(self._chi(uR, eR, entmin), axis=0))
            e = np.minimum(e, np.min(eR, axis=0))

        return rho, p, e, X

    def _get_minima_linearise(self, u, modes, entmin):
        return self._get_minima_bisect(u, modes, entmin)
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

    def _chi(self, u, e, entmin):
        return u[0] * (e - entmin)

    def filter_single(self, umt, ui, f, uidx, entmin):
        pmax = self.order + 1
        v = 1.0
        for p in range(1, pmax):
            v *= v * v * f
            umt[0, p] *= v
            umt[1, p] *= v
            umt[2, p] *= v
        # get new solution
        self.upoly.evaluate(ui, self.efvdm[uidx : uidx + 1], umt)

        rho = ui[0]
        v = ui[1] / rho
        rhoE = ui[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = self.entropy(ui)

        return rho[0], p[0], e[0], self._chi(ui, e, entmin)

    def filter_full(self, umt, unew, f):
        pmax = self.order + 1
        v = 1.0
        for p in range(1, pmax):
            v *= v * v * f
            umt[0, p] *= v
            umt[1, p] *= v
            umt[2, p] *= v
        # get new solution
        self.upoly.evaluate(unew, self.uvdm, umt)

        return

    def _entropy_filter_bisect(self, ubank):
        # assumes entmin_int is already populated
        u = getattr(self, f"u{ubank}")
        # compute solution poly'l modes
        self.ppoly.compute_coeff(self.ua, u, self.invuvdm)

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
            e_tol = 1e-6
        try:
            f_tol = self.config["f_tol"]
        except KeyError:
            f_tol = 1e-4

        # get min entropy for element and neighbors
        entmin = np.min(self.entmin_int, axis=0)

        # compute rho, p, e for all elements
        dmin, pmin, emin, Xmin = self.get_minima(u, self.ua, entmin)

        filtidx = np.where(
            np.bitwise_or(
                dmin < d_min,
                np.bitwise_or(
                    pmin < p_min,
                    Xmin < -e_tol,
                ),
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
                d, p, e, X = self.filter_single(
                    np.copy(umodes), ui, f, uidx, entmin[idx]
                )

                if d < d_min or p < p_min or X < -e_tol:

                    # Setup root finding interval
                    flow = 0.0
                    fhigh = f

                    # Iterate on filter strength
                    for i in range(self.config["efniter"]):

                        # define new f
                        f = 0.5 * (flow + fhigh)

                        d, p, e, X = self.filter_single(
                            np.copy(umodes), ui, f, uidx, entmin[idx]
                        )

                        if d < d_min or p < p_min or X < -e_tol:
                            fhigh = f
                        else:
                            flow = f

                        if fhigh - flow < f_tol:
                            break

                    f = flow

            umodes = self.ua[:, :, idx : idx + 1]
            ## Filter entire solution with flow
            self.filter_full(np.copy(umodes), unew, f)

            # Update final solution with filtered values
            u[:, :, idx : idx + 1] = unew

            # update modes
            self.upoly.compute_coeff(self.ua[:, :, idx : idx + 1], unew, self.invuvdm)

        # update all min interface entropy
        _, _, emin, _ = self.get_minima(u, self.ua, entmin)
        self.entmin_int[:] = emin

    def _entropy_filter_linearise(self, ubank):
        # assumes entmin_int is already populated
        u = getattr(self, f"u{ubank}")
        # compute solution poly'l modes
        self.ppoly.compute_coeff(self.ua, u, self.invuvdm)

        try:
            d_min = self.config["d_min"]
        except KeyError:
            d_min = 1e-6
        try:
            p_min = self.config["rhoe_min"]
        except KeyError:
            p_min = 1e-6
        try:
            e_tol = self.config["e_tol"]
        except KeyError:
            e_tol = 1e-6

        # get min entropy for element and neighbors
        entmin = np.min(self.entmin_int, axis=0)

        # compute rho, p, e for all elements
        dmin, pmin, _, Xmin = self.get_minima(u, self.ua, entmin)

        filtidx = np.where(
            np.bitwise_or(
                dmin < d_min,
                np.bitwise_or(
                    pmin < p_min,
                    Xmin < -e_tol,
                ),
            )
        )[0]

        for idx in filtidx:
            nupts = self.nupts
            invuvdm = self.invuvdm
            umodes = self.ua[:, :, idx : idx + 1]

            if self.fpts_in_upts:
                ui = np.copy(u[:, :, idx : idx + 1])
            else:
                ui = np.zeros((self.nvar, self.nefpts, 1))
                self.upoly.evaluate(ui, self.efvdm, umodes)

            # First test for negative density
            dmin, pmin, _, Xmin = self.get_minima(ui, umodes, entmin[idx])
            if dmin < d_min:
                theta = (umodes[0, 0, 0] - d_min) / max(
                    umodes[0, 0, 0] - dmin, fpdtype_min
                )
                theta = min(1.0, max(theta, 0.0))
                ui[0, :, 0] = umodes[0, 0, 0] + theta * (ui[0, :, 0] - umodes[0, 0, 0])
                self.upoly.compute_coeff(umodes, ui[:, 0:nupts], invuvdm)
                dmin, pmin, _, Xmin = self.get_minima(ui, umodes, entmin[idx])

            # Now test for negative internal energy
            if pmin < p_min:
                rhoeave = umodes[2, 0, 0] - 0.5 * umodes[1, 0, 0] ** 2 / umodes[0, 0, 0]
                pave = (self.config["gamma"] - 1.0) * rhoeave
                theta = (pave - p_min) / max(pave - pmin, fpdtype_min)
                theta = min(1.0, max(theta, 0.0))

                ui[:, 0:nupts, 0] = umodes[:, 0, 0][:, np.newaxis] + theta * (
                    ui[:, 0:nupts, 0] - umodes[:, 0, 0][:, np.newaxis]
                )
                self.upoly.compute_coeff(umodes, ui[:, 0:nupts], invuvdm)
                dmin, pmin, _, Xmin = self.get_minima(ui, umodes, entmin[idx])

            # Finally, test for entropy
            if Xmin < -e_tol:
                Xavg = self._chi(
                    umodes[:, 0, 0], self.entropy(umodes[:, 0, :]), entmin[idx]
                )
                theta = (Xavg + e_tol) / max(Xavg - Xmin, fpdtype_min)
                theta = min(1.0, max(theta, 0.0))

                ui[:, 0:nupts, 0] = umodes[:, 0, 0][:, np.newaxis] + theta * (
                    ui[:, 0:nupts, 0] - umodes[:, 0, 0][:, np.newaxis]
                )
                self.upoly.compute_coeff(umodes, ui[:, 0:nupts], invuvdm)

            # Update solution
            u[:, :, idx : idx + 1] = ui[:, 0:nupts, :]

            # update modes
            self.ua[:, :, idx : idx + 1] = umodes

        # update all min interface entropy
        _, _, emin, _ = self.get_minima(u, self.ua, entmin)
        self.entmin_int[:] = emin

    def _u_to_f_closed(self, ubank):
        u = getattr(self, f"u{ubank}")
        self.uf[:, 0, :] = u[:, 0, :]
        self.uR[:, 1, :] = u[:, 1, :]

    def _u_to_f_open(self, ubank):
        u = getattr(self, f"u{ubank}")
        # REMEMBER, RIGHT interface is LEFT side of element
        # interpolate solution to element faces
        self.uf = np.einsum("fx, vx... -> vf...", self.M0, u)

    def _bc_wall(self, ul, dul, nl, side):
        # ul is the given state, need to determine exterior
        # ur, then compute the common flux
        # nl is outward normal
        ur = ul
        ur[1] = -ul[1]

        f = np.empty((self.nvar, 1))
        if side == "left":
            self.flux.intflux(ul, ur, f)
        else:
            self.flux.intflux(ur, ul, f)

        return f

    def _bc_same(self, ul, dul, nl, side):
        # ul is the given state, need to determine exterior
        # ur, then compute the common flux
        ur = ul

        f = np.empty((self.nvar, 1))
        if side == "left":
            self.flux.intflux(ul, ur, f)
        else:
            self.flux.intflux(ur, ul, f)

        return f

    def _bc_periodic(self, ul=None, dul=None, nl=None, side=None):
        if side == "left":
            ul = self.uf[:, -1, -1]
            ur = self.uf[:, 0, 0]
        else:
            ul = self.uf[:, -1, -1]
            ur = self.uf[:, 0, 0]

        f = self.flux.intflux(ul, ur)

        return f

    def _bc_nscbc_out_p(self, ul, dul, nl, side):
        p_inf = 1.0
        sigma = 0.25

        # flux on face
        f = np.zeros((self.nvar, 1))
        p, v = self.flux.flux(ul, f)

        rho = ul[0]
        rhov = ul[1]
        rhoE = ul[2]

        if side == "left":
            invJac = self.invJac[0]
        else:
            invJac = self.invJac[-1]

        drho = dul[0] * invJac
        drhov = dul[1] * invJac
        drhoE = dul[2] * invJac

        gamma = self.config["gamma"]
        c = np.sqrt(self.config["gamma"] * p / rho)

        K = sigma * c * (1 - (v / c) ** 2) / 1.0

        # spatial derivative (physical)
        dp = (gamma - 1.0) * (drhoE - 0.5 * drho * v**2 - rhov * drhov)
        dv = (drhov - drho * v) / rho

        # wave amplitude velocities
        lam1 = v - c
        lam2 = v
        lam3 = v + c

        L1 = lam1 * (dp - rho * c * dv)
        L2 = lam2 * (c**2 * drho - dp)
        L3 = K * (p - p_inf)

        # now we can compute d
        d1 = 1 / c**2 * (L2 + 0.5 * (L3 + L1))
        d2 = 1 / 2 * (L3 + L1)
        d3 = 1 / (2 * rho * c) * (L3 - L1)

        # now compute dudt
        dudt = np.zeros((self.nvar, 1))
        dudt[0] = -d1
        dudt[1] = -v * d1 + rho * d3
        dudt[2] = -0.5 * v**2 * d1 - d2 / (gamma - 1) + rhov * d3

        if side == "left":
            dfa = self.dfpoly.diff_coeff(self.fa[:, :, 0])
            vdm = self.ldfvdm
            dg = self.gLf
        else:
            dfa = self.dfpoly.diff_coeff(self.fa[:, :, -1])
            vdm = self.rdfvdm
            dg = self.gRf
            pass
        df = np.zeros((self.nvar, 1))
        # derivative of flux at face
        self.dfpoly.evaluate(df, vdm, dfa)
        fc = (-1.0 / invJac * dudt - df + f * dg) / dg

        return fc

    def _bc_nscbc_in_u(self, ul, dul, nl, side):
        v_inf = 0.0
        sigma = 0.25

        # flux on face
        f = np.zeros((self.nvar, 1))
        p, v = self.flux.flux(ul, f)

        rho = ul[0]
        rhov = ul[1]
        rhoE = ul[2]

        if side == "left":
            invJac = self.invJac[0]
        else:
            invJac = self.invJac[-1]

        drho = dul[0] * invJac
        drhov = dul[1] * invJac
        drhoE = dul[2] * invJac

        gamma = self.config["gamma"]
        c = np.sqrt(self.config["gamma"] * p / rho)

        K = sigma * c * (1 - (v / c) ** 2) / 1.0

        # spatial derivative (physical)
        dp = (gamma - 1.0) * (drhoE - 0.5 * drho * v**2 - rhov * drhov)
        dv = (drhov - drho * v) / rho

        # wave amplitude velocities
        lam1 = v - c
        lam2 = v
        lam3 = v + c

        L1 = K * (v - v_inf)
        L2 = 0.0
        L3 = lam1 * (dp + rho * c * dv)

        # now we can compute d
        d1 = 1 / c**2 * (L2 + 0.5 * (L3 + L1))
        d2 = 1 / 2 * (L3 + L1)
        d3 = 1 / (2 * rho * c) * (L3 - L1)

        # now compute dudt
        dudt = np.zeros((self.nvar, 1))
        dudt[0] = -d1
        dudt[1] = -v * d1 + rho * d3
        dudt[2] = -0.5 * v**2 * d1 - d2 / (gamma - 1) + rhov * d3

        if side == "left":
            dfa = self.dfpoly.diff_coeff(self.fa[:, :, 0])
            vdm = self.ldfvdm
            dg = self.gLf
        else:
            dfa = self.dfpoly.diff_coeff(self.fa[:, :, -1])
            vdm = self.rdfvdm
            dg = self.gRf
            pass
        df = np.zeros((self.nvar, 1))
        # derivative of flux at face
        self.dfpoly.evaluate(df, vdm, dfa)
        fc = (-1.0 / invJac * dudt - df + f * dg) / dg

        return fc

    def _bc_ent_wall(self, ul, side):
        ur = ul
        ur[1] = -ul[1]
        e = self.entropy(ur)
        if side == "left":
            self.entmin_int[0, 0] = min(e[0], self.entmin_int[0, 0])
        else:
            self.entmin_int[1, -1] = min(e[0], self.entmin_int[1, -1])

    def _bc_ent_same(self, ul, side):
        ur = ul
        e = self.entropy(ur)
        if side == "left":
            self.entmin_int[0, 0] = min(e[0], self.entmin_int[0, 0])
        else:
            self.entmin_int[1, -1] = min(e[0], self.entmin_int[1, -1])

    def _bc_ent_periodic(self, *args, **kwargs):
        self.entmin_int[0, 0] = self.entmin_int[1, -1] = min(
            self.entmin_int[0, 0], self.entmin_int[1, -1]
        )

    def RHS(self, ubankin, fbankout):

        u = getattr(self, f"u{ubankin}")
        f = getattr(self, f"u{fbankout}")

        self.entropy_filter(ubankin)

        # interpolate solution to face
        self.u_to_f(ubankin)

        # compte interface entropy
        self.intcent()
        self.bcentl(self.uf[:, 0, 0], "left")
        self.bcentr(self.uf[:, -1, -1], "right")

        # compute pointwise fluxes at solution points
        self.flux.flux(u, f)

        # compute common fluxes at interior faces
        self.fc[:, 1, 0:-1] = self.fc[:, 0, 1::] = self.flux.intflux(
            self.uf[:, -1, 0:-1], self.uf[:, 0, 1::]
        )

        # # compute solution derivative (w.r.t comp coords) on left and right faces
        # duaL = self.dppoly.diff_coeff(self.ua[:, 0])
        # self.dppoly.evaluate(self.duL, self.ldfvdm, duaL)
        # duaR = self.dppoly.diff_coeff(self.ua[:, -1])
        # self.dppoly.evaluate(self.duR, self.rdfvdm, duaR)

        # compute common flux on boundary
        self.fc[:, 0, 0] = self.bcl(self.uf[:, 0, 0], self.duL, -1, "left")
        self.fc[:, -1, -1] = self.bcr(self.uf[:, -1, -1], self.duR, 1, "right")

        # evaluate discontinuous flux at flux points
        self.ff[:] = np.einsum("ux...,vx...->vu...", self.M2, f)

        # Begin building of negdivconf
        negdivconf = getattr(self, f"u{fbankout}")

        # compute flux derivative at solution points
        negdivconf[:] = np.einsum("ux...,vx...->vu...", self.M1, f)

        # add the left/right jumps to negdivconf
        negdivconf += np.einsum("vf...,fx...->vx...", self.fc - self.ff, self.M3)

        # transform to neg flux in physical coords
        negdivconf *= -self.invJac

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
        self.u_to_f(0)

        self.entropy_local(0)
        self.intcent()

        self.bcentl(self.uf[:, 0, 0], "left")
        self.bcentr(self.uf[:, 0, -1], "right")
        self.entropy_filter(0)

    def run(self):
        while self.t < self.config["tend"]:
            try:
                if self.niter % self.config["nout"] == 0:
                    if self.config["outfname"] is not None:
                        fname = f"{self.config["outfname"]}_{self.niter:06d}.png"
                    else:
                        fname = None
                    plot(self, fname)
            except ZeroDivisionError:
                pass
            self.intg.step(self, self.config["dt"])


if __name__ == "__main__":
    config = {
        "p": 2,
        "quad": "gauss-legendre",
        "intg": "rk4",
        "intflux": "rusanov",
        "gamma": 1.4,
        "nout": 0,
        "bcl": "periodic",
        "bcr": "periodic",
        "mesh": "mesh-50.npy",
        "outfname": None,
        "efilt": None,
        "efniter": 20,
    }

    a = system(config)

    x = a.x

    # line
    # rho = 1.0
    # rhov = 1 + x
    # v = rhov / rho
    # rhoE = 1 + x
    # p = (1.4 - 1.0) * (rhoE - 0.5 * rho * v**2)

    # sin
    rho = 2 + np.sin(2 * np.pi * x)
    v = 1.0
    p = 1.0

    # wave
    # center = 0.5
    # height = 0.25
    # rho = 1.0
    # v = 0  # 1 + height * np.exp(-((x - center) ** 2) / (2 * 0.05**2))
    # p = 1 + height * np.exp(-((x - center) ** 2) / (2 * 0.05**2))

    # compute CFL = 0.1
    CFL = 0.1
    dx = 1.0 / a.neles / (config["p"] + 1)
    gamma = a.config["gamma"]
    c = np.sqrt(gamma * np.max(p) / np.min(rho)) + np.max(np.abs(v))
    dt = CFL * dx / c
    config["dt"] = dt
    config["tend"] = 5.0

    a.set_ics([rho, v, p])
    a.run()
    plot(a)

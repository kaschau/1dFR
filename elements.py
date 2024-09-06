import numpy as np
from poly import LegendrePoly
from integrators import BaseIntegrator
from flux import BaseFlux
from util import subclass_where
from pathlib import Path
from output import plot

# np.seterr(all="raise")
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
    from poly import LegendrePoly

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
    def __init__(self, config, ics):
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
        self.invvudm = np.linalg.inv(self.uvdm)

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
        # solution point fluxes
        self.f = np.zeros((nvar, nupts, neles))
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
        # dudt physical
        self.negdivconf = np.zeros((nvar, nupts, neles))

        # create integrator
        self.intg = subclass_where(BaseIntegrator, name=config["intg"])()
        for bank in range(self.intg.nbanks):
            setattr(self, f"u{bank}", np.zeros((nvar, nupts, neles)))

        # see if we are filtering
        self.efilt = self.config["efilt"]
        if self.efilt and self.order > 0:
            self.entmin_int = np.zeros(neles + 1)
            self.entropy = getattr(self, f"_entropy_{config["effunc"]}")
            self.entropy_local = self._entropy_local
            self.entropy_filter = self._entropy_filter
            self.bcent = self._bcent
        else:
            self.entropy = noop
            self.entropy_local = noop
            self.entropy_filter = noop
            self.bcent = noop

        # last, set ics
        self.set_ics(ics)

    def _entropy_physical(self, u):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        gamma = self.config["gamma"]
        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        e = np.ones(p.shape)*fpdtype_max
        idx = np.where(np.bitwise_and(rho > 0.0, p > 0.0))
        e[idx] = rho[idx]*(np.log(p[idx]) - gamma*np.log(rho[idx]))

        return e

    def _entropy_local(self, ubank):
        # assume ubank are current
        u = getattr(self, f"u{ubank}")

        #compute element entropy
        sele = np.min(self.entropy(u), axis=0)

        #compute min for elements with right neighbors
        self.entmin_int[0:-1] = sele

        #compute min for elements with left neighbors
        self.entmin_int[1::] = np.minimum(sele, self.entmin_int[0:-1])

        #compute interface entropy
        if not self.fpts_in_upts:
            self.u_to_f()
            self.entmin_int = np.minimum(np.min(self.entropy(self.uL), axis=0), self.entmin_int)
            self.entmin_int = np.minimum(np.min(self.entropy(self.uR), axis=0), self.entmin_int)

    def get_minima(self, u, modes):
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
            #interpolate to faces
            uL = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uL, self.rvdm, modes)
            uR = np.zeros((self.nvar, 1, u.shape[-1]))
            self.upoly.evaluate(uR, self.lvdm, modes)

            rhoL = uL[0]
            vL = u[1] / rhoL
            rhoEL = u[2]
            pL = (gamma - 1.0) * (rhoEL - 0.5 * rhoL * vL**2)
            eL = self.entropy(uL)

            rho = np.minimum(rho, np.min(rhoL, axis=0))
            p = np.minimum(p, np.min(pL, axis=0))
            e = np.minimum(e, np.min(eL, axis=0))

            rhoR = uR[0]
            vR = u[1] / rhoR
            rhoER = u[2]
            pR = (gamma - 1.0) * (rhoER - 0.5 * rhoR * vR**2)
            eR = self.entropy(uR)

            rho = np.minimum(rho, np.min(rhoR, axis=0))
            p = np.minimum(p, np.min(pR, axis=0))
            e = np.minimum(e, np.min(eR, axis=0))

        return rho, p ,e

    def filter_single(self, umt, unew, f):
        pmax = self.order + 1
        v = v2 = 1.0
        for p in range(1, pmax):
            v2 *= v*v*f
            v *= f
            umt[:, p] *= v2
        # get new solution
        self.upoly.evaluate(unew, self.uvdm, umt)

        return self.get_minima(unew, umt)

    def _entropy_filter(self, ubank):
        # assumes entmin_int is already populated
        u = getattr(self, f"u{ubank}")

        d_min = 1e-6
        p_min = 1e-6
        e_tol = 1e-6
        f_tol = 1e-6

        # get min entropy for element and neighbors
        e_min = np.minimum(self.entmin_int[0:-1], self.entmin_int[1::])

        # compute rho, p, e for all elements
        dmin, pmin, emin = self.get_minima(u, self.ua)

        filtidx = np.where(np.bitwise_or(dmin < d_min,
                                         np.bitwise_or(pmin < p_min,
                                         emin < e_min - e_tol)))[0]

        for idx in filtidx:
            umodes = self.ua[:,:,idx:idx+1]
            unew = np.copy(u[:,:,idx:idx+1])

            f = 1.0
            flow = 0.0
            fhigh = f

            i = 0
            while (i < self.config["efniter"]) and (fhigh - flow > f_tol):
                i += 1

                ## ##
                ## import matplotlib.pyplot as plt
                ## plt.plot(np.linspace(-1,1,self.nupts),u[0,:,idx], label="OG")
                ## plt.plot(np.linspace(-1,1,self.nupts),unew[0], marker='o', label="new")
                ## plt.plot([-1,1], [umodes[0,0], umodes[0,0]], label="mean")
                ## plt.legend()
                ## plt.show()
                ## ##

                f = 0.5*(flow + fhigh)

                d, p, e = self.filter_single(np.copy(umodes), unew, f)

                if (d < d_min or
                    p < p_min or
                    e < e_min[idx] - e_tol):
                    fhigh = f
                else:
                    flow = f

            # Update final solution with filtered values
            u[:,:,idx:idx+1] = unew

            # update modes
            self.upoly.compute_coeff(self.ua[:, :, idx:idx+1], unew, self.invvudm)

            # update min interface entropy
            self.entmin_int[idx] = min(e[0], self.entmin_int[idx])
            self.entmin_int[idx + 1] = min(e[0], self.entmin_int[idx + 1])

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
        uL = self.uL[:,:,0]
        eL = self.entropy(uL)
        self.entmin_int[0] = min(eL[0], self.entmin_int[0])

        uR = self.uR[:,:,-1]
        eR = self.entropy(uR)
        self.entmin_int[-1] = min(eR[0], self.entmin_int[-1])

    def update_solution_stuff(self, ubank):
        u = getattr(self, f"u{ubank}")
        # create solution poly'l
        self.upoly.compute_coeff(self.ua, u, self.invvudm)

        self.entropy_filter(ubank)
        # update local entropy
        self.entropy_local(ubank)

        # interpolate solution to face
        self.u_to_f(ubank)

        # compute bcs
        self.bc()
        self.bcent()

    def update_flux_stuff(self, ubank):
        u = getattr(self, f"u{ubank}")
        # compute pointwise fluxes
        self.flux.flux(u, self.f)

        # compute flux poly'l coeffs
        self.upoly.compute_coeff(self.fa, self.f, self.invvudm)

        # compute common fluxes
        self.flux.intflux(self.uL, self.uR, self.fc)

    def build_negdivconf(self):
        # Begin building of negdivconf

        # compute flux derivative at solution points
        self.dfa[:] = self.dfpoly.diff_coeff(self.fa)
        self.dfpoly.evaluate(self.negdivconf, self.dfvdm, self.dfa)

        # evaluate + add the left jumps to negdivconf
        self.upoly.evaluate(self.fl, self.lvdm, self.fa)
        self.negdivconf += np.einsum(
            "vx...,x...->vx...", self.fc[:, :, 0:-1] - self.fl, self.gL
        )

        # eval + add the right jumps to negdivconf
        self.upoly.evaluate(self.fr, self.rvdm, self.fa)
        self.negdivconf += np.einsum(
            "vx...,x...->vx...", self.fc[:, :, 1::] - self.fr, self.gR
        )

        # transform to neg flux in physical coords
        self.negdivconf *= -self.invJac

    def RHS(self, ubank):

        # assumes ubank is up to date
        self.update_solution_stuff(ubank)

        self.update_flux_stuff(ubank)

        self.build_negdivconf()

    def read_grid(self):
        fname = config["mesh"]
        with open(fname, 'rb') as f:
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
        u[2, :] = 0.5 * pris[0] * pris[1] ** 2 + pris[2] / (
            self.config["gamma"] - 1.0
        )

        self.upoly.compute_coeff(self.ua, u, self.invvudm)
        self.u_to_f(0)
        self.bc()

        self.entropy_local(0)
        self.bcent()
        self.update_solution_stuff(0)

    def run(self):
        while round(self.t, 5) <= self.config["tend"]:
            if self.niter % config["nout"] == 0:
                plot(a, f"{config["outfname"]}_{a.niter:06d}.png")
            self.intg.step(self, self.config["dt"])


if __name__ == "__main__":
    config = {
        "p": 3,
        "quad": "gauss-legendre",
        "intg": "rk3",
        "intflux": "hllc",
        "gamma": 1.4,
        "nout": round(0.01/1e-4),
        "bc": "wall",
        "mesh": "mesh-100.npy",
        "dt": 1e-4,
        "tend": 0.2,
        "outfname": "oneD",
        "efilt": True,
        "effunc": "physical",
        "efniter": 20,
    }


    neles = int(config['mesh'].split("-")[1].split(".")[0])
    half = int(neles/2)

    rho = np.zeros((config["p"]+1,neles))
    rho[:, 0:half] = 1.0
    rho[:, half::] = 0.125

    v = 0

    p = np.zeros(rho.shape)
    p[:, 0:half] = 1.0
    p[:, half::] = 0.1

    a = system(config, [rho,v,p])

    a.run()

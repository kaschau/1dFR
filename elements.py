import numpy as np
from poly import LegendrePoly
from integrators import BaseIntegrator
from flux import BaseFlux
from util import subclass_where
from pathlib import Path
from output import plot

np.seterr(all="raise")

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
    def __init__(self, config):
        self.config = config
        self.t = 0.0
        self.niter = 0

        self.nvar = nvar = 3
        self.deg = deg = config["p"]
        self.nupts = nupts = deg + 1

        # solution and flux deriv polys
        self.upoly = LegendrePoly(deg)
        self.dfpoly = LegendrePoly(deg - 1)

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
        self.gL, self.gR = vcjg(deg, c, self.upts, der=True)

        # set interpolation to face
        if self.fpts_in_upts:
            self._u_to_f = self._u_to_f_closed
        else:
            self._u_to_f = self._u_to_f_open

        # SET BOUNDARY CONDITIONS
        if config["bc"] == "wall":
            self._bc = self._bc_wall
        elif config["bc"] == "periodic":
            self._bc = self._bc_periodic

        # set flux
        self.flux = subclass_where(BaseFlux, name=config["intflux"])(config)

        # read grid
        self._read_grid()
        neles = self.neles

        # allocate arrays
        # solution modes
        self.ua = np.zeros((nvar, nupts, neles))
        # solution point fluxes
        self.f = np.zeros((nvar, nupts, neles))
        # flux poly'l modes
        self.fa = np.zeros((nvar, nupts, neles))
        # solution @ Xi=-1
        self.uL = np.zeros((nvar, 1, neles + 1))
        # solution @ Xi=1
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

        # set time integration
        self._set_intg()

    def _u_to_f_closed(self):
        # REMEMBER, RIGHT interface is LEFT side of element
        self.uL[:, 0, 1::] = self.soln[:, -1, :]
        self.uR[:, 0, 0:-1] = self.soln[:, 0, :]

    def _u_to_f_open(self):
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

    def _set_intg(self):
        nvar = self.nvar
        nupts = self.nupts
        neles = self.neles
        # create integrator
        self.intg = subclass_where(BaseIntegrator, name=config["intg"])()
        for bank in range(self.intg.nbanks):
            setattr(self, f"u{bank}", np.zeros((nvar, nupts, neles)))

    def _update_solution_stuff(self):
        # create solution poly'l
        self.upoly.compute_coeff(self.ua, self.soln, self.invvudm)

        # interpolate solution to faces
        self._u_to_f()

        # compute bcs
        self._bc()

    def _update_flux_stuff(self):
        # compute pointwise fluxes
        self.flux.flux(self.soln, self.f)

        # compute flux poly'l coeffs
        self.upoly.compute_coeff(self.fa, self.f, self.invvudm)

        # compute common fluxes
        self.flux.intflux(self.uL, self.uR, self.fc)

    def _build_negdivconf(self):
        # Begin building of negdivconf

        # compute flux derivative at solution points
        self.dfa[:] = self.dfpoly.diff_coeff(self.fa)
        self.dfpoly.evaluate(self.negdivconf, self.dfvdm, self.dfa)

        # evaluate + add the left jumps to negdivconf
        self.upoly.evaluate(self.fl, self.lvdm, self.fa)
        self.negdivconf += np.einsum(
            "ij...,j...->ij...", self.fc[:, :, 0:-1] - self.fl, self.gL
        )

        # eval + add the right jumps to negdivconf
        self.upoly.evaluate(self.fr, self.rvdm, self.fa)
        self.negdivconf += np.einsum(
            "ij...,j...->ij...", self.fc[:, :, 1::] - self.fr, self.gR
        )

        # transform to neg flux in physical coords
        self.negdivconf *= -self.invJac

    def _RHS(self, ubank):

        self.soln = getattr(self, f"u{ubank}")

        self._update_solution_stuff()

        self._update_flux_stuff()

        self._build_negdivconf()


    def compute_emin(self):
        pass

    @staticmethod
    def noop(*args, **kwargs):
        pass

    def _read_grid(self):
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
        # density
        self.u0[0, :] = pris[0]
        # momentum
        self.u0[1, :] = pris[1] * pris[0]
        # total energy
        self.u0[2, :] = 0.5 * pris[0] * pris[1] ** 2 + pris[2] / (
            self.config["gamma"] - 1.0
        )

    def run(self):
        while self.t < self.config["tend"]:
            if self.niter % config["nout"] == 0:
                plot(a, f"{config["outfname"]}_{a.niter:06d}.png")
            self.intg.step(self, self.config["dt"])


if __name__ == "__main__":
    config = {
        "p": 4,
        "quad": "gauss-legendre-lobatto",
        "intg": "rk3",
        "intflux": "rusanov",
        "gamma": 1.4,
        "nout": 1000,
        "bc": "periodic",
        "mesh": "mesh.npy",
        "dt": 1e-4,
        "tend": 1.0,
        "outfname": "oneD",
    }
    a = system(config)
    a.set_ics([np.sin(2.0 * np.pi * a.x) + 2.0, 1.0, 1.0])
    a.run()

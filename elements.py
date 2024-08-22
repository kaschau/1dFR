import numpy as np
from pathlib import Path


class system:
    def __init__(self, p, neles, quadrule):
        self.nvar = nvar = 3
        self.deg = deg = p

        # get num solution points
        self.get_quad_rules(p, quadrule)
        if min(self.absic) < -0.99999999:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        self.nupts = nupts = len(self.absic)

        # create grid
        self.x = np.zeros((nupts, neles))
        self.create_grid(neles)

        # create initial conditions
        self.u = np.zeros((nvar, nupts, neles))
        self.set_ics()

        # create inverse/vandermonde matrix
        self.vdm = np.zeros((nupts, nupts))
        self.invdm = np.zeros((nupts, nupts))
        self.vandermonde()

        # create solution poly'l
        self.ua = np.zeros((nvar, nupts, neles))
        self.update_soln_poly()

        # interpolate to faces
        self.uL = np.zeros((nvar, neles))
        self.uR = np.zeros((nvar, neles))
        if self.fpts_in_upts:
            self.interpolate_to_face = self.noop
            self.lvdm = None
            self.rvdm = None
        else:
            self.interpolate_to_face = self.i2f
            self.lvdm = np.polynomial.legendre.legvander([-1], self.deg)
            self.rvdm = np.polynomial.legendre.legvander([1], self.deg)
        self.i2f()

        self.f = np.zeros((nvar, nupts, neles))

    def noop(*args, **kwargs):
        pass

    def get_quad_rules(self, p, rule):
        if "lobatto" in rule:
            fname = f"/quadrules/gauss-legendre-lobatto-n{p+1}-d{2*(p)-1}-spu.txt"
        else:
            fname = f"/quadrules/gauss-legendre-n{p+1}-d{2*p+1}-spu.txt"

        direct = str(Path(__file__).parent)
        with open(direct + fname) as f:
            data = np.genfromtxt(f, delimiter=" ")
            self.absic = data[:, 0]
            self.weights = data[:, 1]

    def create_grid(self, neles):
        eles = np.zeros((neles, 2))
        eles[:, 0] = np.linspace(0, 1, neles + 1)[0:-1]
        eles[:, 1] = np.linspace(0, 1, neles + 1)[1::]
        h = eles[:, 1] - eles[:, 0]
        self.x[:] = np.mean(eles, axis=-1)[np.newaxis, :] + np.einsum(
            "i,j->ij", self.absic, h / 2.0
        )

    def set_ics(self):
        # density
        self.u[0, :] = 1.2
        # momentum
        self.u[1, :] = self.x[:]
        # total energy
        self.u[2, :] = 1.0

    def vandermonde(self):
        self.vdm[:] = np.polynomial.legendre.legvander(self.absic, self.deg)
        self.invdm[:] = np.linalg.inv(self.vdm)

    def update_soln_poly(self):
        self.ua[:] = np.einsum("ji...,ki...->ki...", self.invdm, self.u)

    def i2f(self):
        self.uL[:] = np.einsum("ji...,ki...->k...", self.lvdm, self.ua)
        self.uR[:] = np.einsum("ji...,ki...->k...", self.rvdm, self.ua)

        pass


if __name__ == "__main__":
    p = 3
    neles = 5
    quad = "gauss-legendre"
    a = system(p, neles, quad)

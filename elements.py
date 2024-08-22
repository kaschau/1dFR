import numpy as np
from pathlib import Path

gamma = 1.4


def compute_coeff(a, y, invdm):
    a[:] = np.einsum("ji...,ki...->kj...", invdm, y)


def invflux(u, f):
    rho = u[0]
    v = u[1] / rho
    rhoE = u[2]
    gamma = 1.4

    p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

    f[0] = u[1]
    f[1] = u[1] * v + p
    f[2] = (rhoE + p) * v

    return p, v


def rusanov(uL, uR, f):
    # wavespeed
    fL = np.zeros(f.shape)
    fR = np.zeros(f.shape)

    pL = np.zeros((f.shape[-1]))
    pR = np.zeros((f.shape[-1]))

    vL = np.zeros((f.shape[-1]))
    vR = np.zeros((f.shape[-1]))

    pL[:], vL[:] = invflux(uL, fL)
    pR[:], vR[:] = invflux(uR, fR)

    lam = np.maximum(
        np.abs(uL) + np.sqrt(gamma * pL / uL[0]),
        np.abs(uR) + np.sqrt(gamma * pR / uR[0]),
    )

    f[:] = 0.5 * (fR + fL - lam * (uR - uL))


class system:
    def __init__(self, p, neles, quadrule):
        self.nvar = nvar = 3
        self.deg = deg = p

        # get num solution points
        self.get_quad_rules(p, quadrule)
        if min(self.Xi) < -0.99999999:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        self.nupts = nupts = len(self.Xi)

        # create grid
        self.x = np.zeros((nupts))
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
        compute_coeff(self.ua, self.u, self.invdm)

        # compute pointwise fluxes
        self.f = np.zeros((nvar, nupts, neles))
        invflux(self.u, self.f)

        # compute flux poly'l
        self.fa = np.zeros((nvar, nupts, neles))
        compute_coeff(self.fa, self.f, self.invdm)

        # interpolate to faces
        self.uL = np.zeros((nvar, neles + 1))
        self.uR = np.zeros((nvar, neles + 1))
        if self.fpts_in_upts:
            self.interpolate_to_face = self.noop
            self.lvdm = None
            self.rvdm = None
        else:
            self.interpolate_to_face = self.i2f
            self.lvdm = np.polynomial.legendre.legvander([-1], self.deg)
            self.rvdm = np.polynomial.legendre.legvander([1], self.deg)
        self.i2f()

        # set BCS
        self.uL[:, -1] = self.uL[:, -2]
        self.uR[:, 0] = self.uR[:, 1]

        # compute common fluxes
        self.fc = np.zeros((nvar, neles + 1))
        rusanov(self.uL, self.uR, self.fc)

        # define correction functions
        #gL = np.polynomial.

    def noop(*args, **kwargs):
        pass

    def get_quad_rules(self, p, rule):
        if "lobatto" in rule:
            fname = f"/quadrules/gauss-legendre-lobatto-n{p+1}-d{2*(p)-1}-spu.txt"
        else:
            fname = f"/quadrules/gauss-legendre-n{p+1}-d{2*p+1}-spu.txt"

        # Get Solution Points
        direct = str(Path(__file__).parent)
        with open(direct + fname) as f:
            data = np.genfromtxt(f, delimiter=" ")
            self.Xi = data[:, 0]
            self.weights = data[:, 1]
        # Get correction function points
        direct = str(Path(__file__).parent)
        try:
            fname = f"/quadrules/gauss-legendre-lobatto-n{p+2}-d{2*(p+1)-1}-spu.txt"
            with open(direct + fname) as f:
                data = np.genfromtxt(f, delimiter=" ")
                self.Zeta = data[:, 0]
        except FileNotFoundError:
            self.Zeta = np.array([-1, 1])

    def create_grid(self):
        neles = self.neles
        eles = np.zeros((neles, 2))
        eles[:, 0] = np.linspace(0, 1, neles + 1)[0:-1]
        eles[:, 1] = np.linspace(0, 1, neles + 1)[1::]
        h = eles[:, 1] - eles[:, 0]
        self.x[:] = np.mean(eles, axis=-1)[np.newaxis, :] + np.einsum(
            "i,j->ij", self.Xi, h / 2.0
        )

    def set_ics(self):
        # density
        self.u[0, :] = 1.2
        # momentum
        self.u[1, :] = self.x[:]
        # total energy
        self.u[2, :] = 1.0

    def vandermonde(self):
        self.vdm[:] = np.polynomial.legendre.legvander(self.Xi, self.deg)
        self.invdm[:] = np.linalg.inv(self.vdm)

    def i2f(self):
        self.uL[:, 0:-1] = np.einsum("ji...,ki...->k...", self.lvdm, self.ua)
        self.uR[:, 1::] = np.einsum("ji...,ki...->k...", self.rvdm, self.ua)


if __name__ == "__main__":
    p = 3
    neles = 5
    quad = "gauss-legendre"
    a = system(p, neles, quad)

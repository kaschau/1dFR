import numpy as np
from poly import LegendrePoly
from pathlib import Path
from matplotlib import pyplot as plt

gamma = 1.4


def get_quad_rules(p, rule):
    if "lobatto" in rule:
        fname = f"/quadrules/gauss-legendre-lobatto-n{p+1}-d{2*(p)-1}-spu.txt"
        direct = str(Path(__file__).parent)
        with open(direct + fname) as f:
            data = np.genfromtxt(f, delimiter=" ")
            return data[:, 0]
    else:
        data = np.polynomial.legendre.leggauss(p + 1)
        return data[0]


def flux(u, f):
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

    pL[:], vL[:] = flux(uL, fL)
    pR[:], vR[:] = flux(uR, fR)

    lam = np.maximum(
        np.abs(uL) + np.sqrt(gamma * pL / uL[0]),
        np.abs(uR) + np.sqrt(gamma * pR / uR[0]),
    )

    f[:] = 0.5 * (fR + fL - lam * (uR - uL))


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
    if not der:
        Lk = Legk.basis_at
        Lkm = Legkm.basis_at
        Lkp = Legkp.basis_at
    else:
        Lk = Legk.dbasis_at
        Lkm = Legkm.dbasis_at
        Lkp = Legkp.dbasis_at

    g = lambda x: 0.5 * (Lk(x) + (etak * Lkm(x) + Lkp(x)) / (1 + etak))
    gr = g(x)
    gl = g(-x)

    return gl, gr


class system:
    def __init__(self, p, neles, solpts):
        self.nvar = nvar = 3
        self.neles = neles
        self.deg = deg = p

        self.upoly = LegendrePoly(deg)
        self.dfpoly = LegendrePoly(deg - 1)

        # get num solution points
        self.upts = get_quad_rules(p, solpts)
        if min(self.upts) < -0.99999999:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        self.nupts = nupts = len(self.upts)

        # create grid
        self.x = np.zeros((nupts, neles))
        self.create_grid()

        # create initial conditions
        self.u = np.zeros((nvar, nupts, neles))
        self.set_ics()

        # create inverse/vandermonde matrix
        self.vdm = np.zeros((nupts, nupts))
        self.invdm = np.zeros((nupts, nupts))
        self.vdm[:] = self.upoly.vandermond(self.upts)
        self.invdm[:] = np.linalg.inv(self.vdm)

        # create solution poly'l
        self.ua = np.zeros((nvar, nupts, neles))
        self.upoly.compute_coeff(self.ua, self.u, self.invdm)

        # compute pointwise fluxes
        self.f = np.zeros((nvar, nupts, neles))
        flux(self.u, self.f)

        # compute flux poly'l
        self.fa = np.zeros((nvar, nupts, neles))
        self.upoly.compute_coeff(self.fa, self.f, self.invdm)

        # Begin building of negdivconf
        self.negdifconv = np.zeros((nvar, nupts, neles))
        # compute flux derivative at solution points
        dfa = self.dfpoly.diff_coeff(self.fa)
        self.dfpoly.evaluate(self.negdifconv,
                             self.dfpoly.vandermond(self.upts),
                             dfa)

        # interpolate solution to faces
        self.uL = np.zeros((nvar, neles + 1))
        self.uR = np.zeros((nvar, neles + 1))
        if not self.fpts_in_upts:
            # interpolate solution to left face
            lvdm = self.upoly.vandermond([-1])
            self.upoly.evaluate(self.uL[:, 1::], lvdm, self.ua)
            # interpolate solution to right face
            rvdm = self.upoly.vandermond([1])
            self.upoly.evaluate(self.uR[:, 0:-1], rvdm, self.ua)

        # SET BOUNDARY CONDITIONS

        # walls
        # self.uL[:, 0] = self.uR[:, 0]
        # self.uR[:, -1] = self.uL[:, -1]

        # periodics
        self.uL[:, 0] = self.uL[:, -1]
        self.uR[:, -1] = self.uL[:, -1]

        # compute common fluxes
        self.fc = np.zeros((nvar, neles + 1))
        rusanov(self.uL, self.uR, self.fc)

        # compute g' of correction functions at solution points
        c = 0  # Vincent constant 0 = nodal DG
        self.gL, self.gR = vcjg(deg, c, self.upts, der=True)

    def noop(*args, **kwargs):
        pass

    def create_grid(self):
        neles = self.neles
        eles = np.zeros((neles, 2))
        eles[:, 0] = np.linspace(0, 1, neles + 1)[0:-1]
        eles[:, 1] = np.linspace(0, 1, neles + 1)[1::]
        h = eles[:, 1] - eles[:, 0]
        self.x[:] = np.mean(eles, axis=-1)[np.newaxis, :] + np.einsum(
            "i,j->ij", self.upts, h / 2.0
        )

    def set_ics(self):
        # density
        self.u[0, :] = 1.0
        # momentum
        self.u[1, :] = 1.0
        # total energy
        self.u[2, :] = 1.0

    def i2f(self):
        self.uL[:, 1::] = np.einsum("ji...,ki...->k...", self.lvdm, self.ua)
        self.uR[:, 0:-1] = np.einsum("ji...,ki...->k...", self.rvdm, self.ua)


if __name__ == "__main__":
    p = 3
    neles = 5
    quad = "gauss-legendre"
    a = system(p, neles, quad)

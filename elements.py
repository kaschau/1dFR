import numpy as np
from poly import LegendrePoly
from integrators import BaseIntegrator
from util import subclass_where
from pathlib import Path
from matplotlib import pyplot as plt

gamma = 1.4
np.seterr(all='raise')


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

    f[:] = 0.5 * (fL + fR - lam * (uR - uL))


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
    def __init__(self, p, solpts):
        self.t = 0.0
        self.niter = 0

        self.nvar = nvar = 3
        self.deg = deg = p
        self.neles = neles
        self.nupts = nupts = p + 1

        self.upoly = LegendrePoly(deg)
        self.dfpoly = LegendrePoly(deg - 1)

        # get num solution points
        self.upts = get_quad_rules(p, solpts)
        if min(self.upts) < -0.99999999:
            self.fpts_in_upts = True
        else:
            self.fpts_in_upts = False

        # create solution point inverse/vandermonde matrix
        self.uvdm = self.upoly.vandermonde(self.upts)
        self.invudm = np.linalg.inv(self.uvdm)

        # left and right solution vandermonde
        self.lvdm = self.upoly.vandermonde([-1])
        self.rvdm = self.upoly.vandermonde([1])

        # flux derivative vandermonde
        self.dfvdm = self.dfpoly.vandermonde(self.upts)

        # compute g' of correction functions at solution points
        c = 0  # Vincent constant 0 = nodal DG
        self.gL, self.gR = vcjg(deg, c, self.upts, der=True)

        # xt = np.linspace(-1, 1, 100)
        # gL, gR = vcjg(deg, c, xt, der=False)
        # plt.plot(xt, gL, label="l")
        # plt.plot(xt, gR, label="r")
        # dgL, dgR = vcjg(deg, c, xt, der=True)
        # plt.plot(xt, dgL, label="dl")
        # plt.plot(xt, dgR, label="dr")
        # plt.legend()
        # plt.grid(visible=True)
        # plt.show()

    def set_RHS(self):
        nvar = self.nvar
        neles = self.neles
        nupts = self.nupts
        # RHS arrays
        # solution modes
        self.ua = np.zeros((nvar, nupts, neles))
        # solution point fluxes
        self.f = np.zeros((nvar, nupts, neles))
        # flux poly'l modes
        self.fa = np.zeros((nvar, nupts, neles))
        # solution @ eta=-1
        self.uL = np.zeros((nvar, 1, neles + 1))
        # solution @ eta=1
        self.uR = np.zeros((nvar, 1, neles + 1))
        # continuous flux values
        self.fc = np.zeros((nvar, 1, neles + 1))
        # flux derivative modes
        self.dfa = np.zeros((nvar, nupts - 1, neles))
        # flux polyl @ eta=-1
        self.fl = np.zeros((nvar, 1, neles))
        # flux polyl @ eta=1
        self.fr = np.zeros((nvar, 1, neles))
        # dudt physical
        self.negdivconf = np.zeros((nvar, nupts, neles))

    def set_intg(self, intg):
        nvar = self.nvar
        nupts = self.nupts
        # create integrator
        self.intg = subclass_where(BaseIntegrator, name=intg)()
        for bank in range(self.intg.nbanks):
            setattr(self, f"u{bank}", np.zeros((nvar, nupts, neles)))
        self.u = self.u0

    def RHS(self, ubank):

        soln = getattr(self, f"u{ubank}")

        # create solution poly'l
        self.upoly.compute_coeff(self.ua, soln, self.invudm)

        # compute pointwise fluxes
        flux(soln, self.f)

        # compute flux poly'l
        self.upoly.compute_coeff(self.fa, self.f, self.invudm)

        # interpolate solution to faces
        # REMEMBER, RIGHT interface is LEFT side of element
        if not self.fpts_in_upts:
            # interpolate solution to left face
            self.upoly.evaluate(self.uL[:, :, 1::], self.rvdm, self.ua)
            # interpolate solution to right face
            self.upoly.evaluate(self.uR[:, :, 0:-1], self.lvdm, self.ua)
        else:
            self.uL[:, :, 1::] = soln[:, -1, :]
            self.uR[:, :, 0:-1] = soln[:, 0, :]

        # SET BOUNDARY CONDITIONS
        if self.bc == "wall":
            self.uL[:, :, 0] = self.uL[:, :, 0]
            self.uR[:, :, -1] = self.uL[:, :, -1]
        elif self.bc == "periodic":
            self.uL[:, :, 0] = self.uL[:, :, -1]
            self.uR[:, :, -1] = self.uR[:, :, 0]

        # compute common fluxes
        rusanov(self.uL, self.uR, self.fc)

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

    def noop(*args, **kwargs):
        pass

    def create_grid(self, neles):
        eles = np.zeros((neles, 2))
        eles[:, 0] = np.linspace(0, 1, neles + 1)[0:-1]
        eles[:, 1] = np.linspace(0, 1, neles + 1)[1::]
        h = eles[:, 1] - eles[:, 0]
        self.x = np.mean(eles, axis=-1)[np.newaxis, :] + np.einsum(
            "i,j->ij", self.upts, h / 2.0
        )
        self.invJac = 2.0 / h

    def set_ics(self, pris):
        # density
        self.u[0, :] = pris[0]
        # momentum
        self.u[1, :] = pris[1] * pris[0]
        # total energy
        self.u[2, :] = 0.5 * pris[0] * pris[1] ** 2 + pris[2] / (gamma - 1.0)

    def set_bcs(self, bc):
        self.bc = bc

    def plot(self):
        rho = self.u[0]
        v = self.u[1] / rho
        rhoE = self.u[2]

        p = (gamma - 1.0) * (rhoE - 0.5 * rho * v**2)

        plt.plot(self.x.ravel(order="F"), self.u[0].ravel(order="F"), label="rho", marker='o')
        plt.plot(self.x.ravel(order="F"), p.ravel(order="F"), label="p", marker="o")
        plt.plot(self.x.ravel(order="F"), v.ravel(order="F"), label="v", marker="o")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    p = 4
    neles = 20
    quad = "gauss-legendre"
    intg = "rk1"
    a = system(p, quad)

    a.create_grid(neles)
    a.set_bcs("periodic")
    a.set_intg(intg)
    a.set_RHS()

    a.set_ics([np.sin(2.0 * np.pi * a.x) + 2.0, 1.0, 1.0])
    # a.set_ics([1.0, 1.0, 1.0])
    a.plot()

    dt = 1e-4
    niter = 140
    while a.t < 0.5:
    # while a.niter < niter:
        a.intg.step(a, dt)
    print(a.t, a.niter)
    a.plot()
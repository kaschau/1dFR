import numpy as np


class BaseFlux:
    def __init__(self, config):
        self.config = config

    def flux(self, u, f):
        rho = u[0]
        v = u[1] / rho
        rhoE = u[2]

        p = (self.config["gamma"] - 1.0) * (rhoE - 0.5 * rho * v**2)

        f[0] = u[1]
        f[1] = u[1] * v + p
        f[2] = (rhoE + p) * v

        return p, v


class Rusanov(BaseFlux):
    name = "rusanov"

    def __init__(self, config):
        super().__init__(config)

    def intflux(self, uL, uR, f):
        fL = np.zeros(f.shape)
        fR = np.zeros(f.shape)

        pL, vL = self.flux(uL, fL)
        pR, vR = self.flux(uR, fR)

        # wavespeed
        gamma = self.config["gamma"]
        a = np.sqrt((0.25 * gamma) * (pL + pR) / (uL[0] + uR[0])) + 0.25 * (
            np.abs(vL + vR)
        )

        f[:] = 0.5 * (fL + fR) - a * (uR - uL)


class HLLC(BaseFlux):
    name = "hllc"

    def __init__(self, config):
        super().__init__(config)

    def intflux(self, uL, uR, f):
        gamma = self.config["gamma"]

        rhol = uL[0, 0]
        rhor = uR[0, 0]

        fl = np.zeros(f.shape)
        fr = np.zeros(f.shape)

        pl = np.zeros((f.shape[-1]))
        pr = np.zeros((f.shape[-1]))

        vl = np.zeros((f.shape[-1]))
        vr = np.zeros((f.shape[-1]))

        pl[:], vl[:] = self.flux(uL, fl)
        pr[:], vr[:] = self.flux(uR, fr)

        rhoEl = uL[2, 0]
        rhoEr = uR[2, 0]

        # Roe ave H
        H = (np.sqrt(rhol) * (pr + rhoEr) + np.sqrt(rhor) * (pl + rhoEl)) / (
            np.sqrt(rhol) * rhor + np.sqrt(rhor) * rhol
        )

        # Roe ave speed of sound
        u = (np.sqrt(rhol) * vl + np.sqrt(rhor) * vr) / (np.sqrt(rhol) + np.sqrt(rhor))
        a = np.sqrt((gamma - 1) * (H - 0.5 * u * u))

        # Estimate l and r wave speed
        sl = u - a
        sr = u + a
        sstar = (pr - pl + rhol * vl * (sl - vl) - rhor * vr * (sr - vr)) / (
            rhol * (sl - vl) - rhor * (sr - vr)
        )

        # Star state factors
        ul_com = (sl - vl) / (sl - sstar)
        ur_com = (sr - vr) / (sr - sstar)

        usl = np.zeros((3, ul_com.shape[0]))
        usr = np.zeros((3, ur_com.shape[0]))

        # star state mass
        usl[0] = ul_com * rhol
        usr[0] = ur_com * rhor

        # star state momentum
        usl[1] = ul_com * rhol * sstar
        usr[1] = ur_com * rhor * sstar

        # star state energy
        usl[2] = ul_com * (rhoEl + (sstar - vl) * (rhol * sstar + pl / (sl - vl)))
        usr[2] = ur_com * (rhoEr + (sstar - vr) * (rhor * sstar + pr / (sr - vr)))

        # output
        idxL = np.where(sl >= 0.0)[0]
        idxLs = np.where(np.bitwise_and(sl <= 0.0, sstar >= 0.0))[0]
        idxRs = np.where(np.bitwise_and(sr >= 0.0, sstar <= 0.0))[0]
        idxR = np.where(sr <= 0.0)[0]
        for i in range(3):
            f[i, 0, idxL] = fl[i, 0, idxL]

            fsl = fl[i, 0, idxLs] + sl[idxLs] * (usl[i, idxLs] - uL[i, 0, idxLs])
            f[i, 0, idxLs] = fsl

            fsr = fr[i, 0, idxRs] + sr[idxRs] * (usr[i, idxRs] - uR[i, 0, idxRs])
            f[i, 0, idxRs] = fsr

            f[i, 0, idxR] = fr[i, 0, idxR]


class exact(BaseFlux):
    name = "exact"

    def __init__(self, config):
        super().__init__(config)

    def intflux(self, uL, uR, f):
        gamma = self.config["gamma"]

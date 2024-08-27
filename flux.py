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

        pL = np.zeros((f.shape[-1]))
        pR = np.zeros((f.shape[-1]))

        vL = np.zeros((f.shape[-1]))
        vR = np.zeros((f.shape[-1]))

        pL[:], vL[:] = self.flux(uL, fL)
        pR[:], vR[:] = self.flux(uR, fR)

        # wavespeed
        lam = np.maximum(
            np.abs(uL) + np.sqrt(self.config["gamma"] * pL / uL[0]),
            np.abs(uR) + np.sqrt(self.config["gamma"] * pR / uR[0]),
        )

        f[:] = 0.5 * (fL + fR - lam * (uR - uL))


class HLLC(BaseFlux):
    name = "hllc"
    def __init__(self, config):
        super().__init__(config)

    def intflux(self, uL, uR, f):
        pass
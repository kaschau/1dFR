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
        gamma = self.config["gamma"]
        self.hgm = 0.5 * (gamma - 1)
        self.grgm = gamma / (gamma - 1)
        self.trgm = 2 / (gamma - 1)
        self.trgp = 2 / (gamma + 1)
        self.gmrtg = (gamma - 1) / (2 * gamma)
        self.gprtg = (gamma + 1) / (2 * gamma)
        self.tgrgm = (2 * gamma) / (gamma - 1)
        self.gmrgp = (gamma - 1) / (gamma + 1)
        self.p_min = 1e-8

    def star_flux(self, p, ps, rs, cs):

        # condition 1
        pr = p / ps
        pr_g = pr**-self.gprtg
        fd1 = pr_g / (rs * cs)
        f1 = self.trgm * cs * (pr_g * pr - 1)

        pbs_inv = 1 / (p + self.gmrgp * ps)
        sapb = np.sqrt(self.trgp * pbs_inv / rs)
        f2 = (p - ps) * sapb
        fd2 = sapb - 0.5 * f2 * pbs_inv

        return np.where(p <= ps, f1, f2), np.where(p <= ps, fd1, fd2)

    def intflux(self, uL, uR, f):
        gamma = self.config["gamma"]
        hgm = 0.5 * (gamma - 1)
        grgm = gamma / (gamma - 1)
        trgm = 2 / (gamma - 1)
        trgp = 2 / (gamma + 1)
        gmrtg = (gamma - 1) / (2 * gamma)
        gprtg = (gamma + 1) / (2 * gamma)
        tgrgm = (2 * gamma) / (gamma - 1)
        gmrgp = (gamma - 1) / (gamma + 1)
        p_min = self.p_min

        rl = uL[0, 0]
        rr = uR[0, 0]
        vl = uL[1, 0] / rl
        vr = uR[1, 0] / rr

        pl = (gamma - 1.0) * (uL[2, 0] - 0.5 * rl * vl**2)
        pr = (gamma - 1.0) * (uR[2, 0] - 0.5 * rr * vr**2)

        cl = np.sqrt(gamma * pl / rl)
        cr = np.sqrt(gamma * pr / rr)

        # Initial pressure guess
        bpv = np.maximum(0, 0.125 * (vl - vr) * (rl + rr) * (cl + cr) + 0.5 * (pl + pr))
        pmin = np.minimum(pl, pr)
        pmax = np.maximum(pl, pr)

        c1 = np.bitwise_and(pmax <= 2 * pmin, np.bitwise_and(pmin <= bpv, bpv <= pmax))
        c2 = np.less(bpv, pmin)

        # condition 1
        p0_1 = bpv
        # condition 2 two rare fractions
        pre = np.power(pl / pr, gmrtg)
        um = (pre * vl * cr + vr * cl + trgm * (pre - 1) * cl * cr) / (pre * cr + cl)
        ptl = 1 - hgm * (um - vl) / cl
        ptr = 1 + hgm * (um - vr) / cr

        p0_2 = 0.5 * (pl * ptl**tgrgm + pr * ptr**tgrgm)

        # two shock case
        gl = np.sqrt(trgp / (rl * (gmrgp * pl + bpv)))
        gr = np.sqrt(trgp / (rr * (gmrgp * pr + bpv)))

        p0_3 = (gl * pl + gr * pr - vr + vl) / (gl + gr)

        p0 = np.where(c1, p0_1, np.where(c2, p0_2, p0_3))

        kmax = 3
        for k in range(kmax):
            fsl, fdl = self.star_flux(p0, pl, rl, cl)
            fsr, fdr = self.star_flux(p0, pr, rr, cr)
            p1 = p0 - (fsl + fsr + vr - vl) / (fdl + fdr)
            p0 = np.where(p1 <= 0, p_min, p1)

        uss = 0.5 * (vl + vr + fsr - fsl)

        w0 = np.empty((3, uss.shape[0]))
        for i, us in enumerate(uss):
            p0i = p0[i]
            pli = pl[i]
            pri = pr[i]

            rli = rl[i]
            rri = rr[i]

            vli = vl[i]
            vri = vr[i]

            cli = cl[i]
            cri = cl[i]

            if 0 <= us:
                w0[1, i] = vli
                if p0i <= pli:
                    if vli >= cli:
                        w0[0, i] = rli
                        w0[1, i] = vli
                        w0[2, i] = pli
                    else:
                        if us < cli * (p0i / pli) ** gmrtg:
                            w0[0, i] = rli * (p0i / pli) ** (1 / gamma)
                            w0[1, i] = us
                            w0[2, i] = p0i
                        else:
                            c = trgp + gmrgp * vli / cli
                            w0[0, i] = rli * c**trgm
                            w0[1, i] = trgp * cli + gmrgp * vli
                            w0[2, i] = pli * c**tgrgm
                else:
                    p0p = p0i / pli
                    sl = vli - cli * np.sqrt(gprtg * p0p + gmrtg)
                    if 0 <= sl:
                        w0[0, i] = rli
                        w0[1, i] = vli
                        w0[2, i] = pli
                    else:
                        w0[0, i] = rli * (p0p + gmrgp) / (p0p * gmrgp + 1)
                        w0[1, i] = us
                        w0[2, i] = p0i
            else:
                w0[1, i] = vri
                if p0i > pri:
                    p0p = p0i / pri
                    sr = vri - cri * np.sqrt(gprtg * p0p + gmrtg)
                    if sr <= 0:
                        w0[0, i] = rri
                        w0[1, i] = vri
                        w0[2, i] = pri
                    else:
                        w0[0, i] = rri * (p0p + gmrgp) / (p0p * gmrgp + 1)
                        w0[1, i] = us
                        w0[2, i] = p0i
                else:
                    if vri + cri <= 0:
                        w0[0, i] = rri
                        w0[1, i] = vri
                        w0[2, i] = pri
                    else:
                        p0p = p0i / pri
                        if us + cri * p0p**gmrtg >= 0:
                            w0[0, i] = rri * (p0p) ** (1 / gamma)
                            w0[1, i] = us
                            w0[2, i] = p0i
                        else:
                            c = trgp - gmrgp * vri / cri
                            w0[0, i] = rri * c**trgm
                            w0[1, i] = gmrgp * vri - trgp * cri
                            w0[2, i] = pri * c**tgrgm

        w0 = np.where(np.abs(w0) < 2.2e-16, 0.0, w0)
        f[0, 0, :] = w0[0] * w0[1]
        f[1, 0, :] = f[0, 0, :] * w0[1] + w0[2]
        f[2, 0, :] = (grgm * w0[2] + 0.5 * w0[0] * w0[1] ** 2) * w0[1]

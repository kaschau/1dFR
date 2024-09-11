#!/usr/bin/env python3

"""
Solves the shock tube problem defined with arbutrary left and right states states separated by a membrane
at some x location between zero and one.

Solves the problem numerically with FR, and analytically using an exact Riemann solver.

See

Riemann Solvers and Numerical Methods for Fluid Dynamic 3rd Ed.
Eleuterio F. Toro
Springer

for more.
"""
import numpy as np
from oneDFR.elements import system
import matplotlib.pyplot as plt
from pathlib import Path

np.seterr(all="raise")

def guessP(test):
    pL, rhoL, uL = test.pL, test.rhoL, test.uL
    pR, rhoR, uR = test.pR, test.rhoR, test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma

    # Gamma constants
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g3 = 2.0 * gamma / (gamma - 1.0)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = (gamma - 1.0) / 2.0

    qUser = 2.0
    cup = 0.25 * (rhoL + rhoR) * (cL + cR)
    ppv = 0.5 * (pL + pR) + 0.5 * (uL - uR) * cup
    ppv = max(0.0, ppv)
    pmin = min(pL, pR)
    pmax = max(pL, pR)
    qmax = pmax / pmin

    if (qmax < qUser) and ((pmin < ppv) and (ppv < pmax)):
        pM = ppv
    else:
        if ppv < pmin:
            pQ = (pL / pR) ** g1
            uM = (pQ * uL / cL + uR / cR + g4 * (pQ - 1.0)) / (pQ / cL + 1.0 / cR)
            pTL = 1.0 + g7 * (uL - uM) / cL
            pTR = 1.0 + g7 * (uM - uR) / cR
            pM = 0.5 * (pL * pTL**g3 + pR * pTR**g3)
        else:
            gEL = np.sqrt((g5 / rhoL) / (g6 * pL + ppv))
            gER = np.sqrt((g5 / rhoR) / (g6 * pR + ppv))
            pM = (gEL * pL + gER * pR - (uR - uL)) / (gEL + gER)

    return pM


def prefun(p, test, side):
    if side == "L":
        pK, rhoK = test.pL, test.rhoL
        cK = test.cL
    else:
        pK, rhoK = test.pR, test.rhoR
        cK = test.cR
    gamma = test.gamma
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)

    if p < pK:
        pRatio = p / pK
        F = g4 * cK * (pRatio**g1 - 1.0)
        FD = (1.0 / (rhoK * cK)) * pRatio ** (-g2)
    else:
        AK = g5 / rhoK
        BK = g6 * pK
        qrt = np.sqrt(AK / (BK + p))
        F = (p - pK) * qrt
        FD = (1.0 - 0.5 * (p - pK) / (BK + p)) * qrt

    return F, FD


def pStar(test):
    uL = test.uL
    uR = test.uR

    maxIter = 100
    tol = 1e-6

    n = 0
    deltaP = 1e10

    pOld = guessP(test)
    uDiff = uR - uL
    for n in range(maxIter):
        fL, fDL = prefun(pOld, test, "L")
        fR, fDR = prefun(pOld, test, "R")
        p = pOld - (fL + fR + uDiff) / (fDL + fDR)
        deltaP = 2.0 * abs((p - pOld) / (p + pOld))
        pOld = p
        if deltaP < tol:
            break
    else:
        raise ValueError("Did not converge.")

    u = 0.5 * (uL + uR + fR - fL)
    return p, u


def sample(test, pM, uM, s):
    pL, rhoL, uL = test.pL, test.rhoL, test.uL
    pR, rhoR, uR = test.pR, test.rhoR, test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma
    # Gamma constants
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)
    g3 = 2.0 * gamma / (gamma - 1.0)
    g4 = 2.0 / (gamma - 1.0)
    g5 = 2.0 / (gamma + 1.0)
    g6 = (gamma - 1.0) / (gamma + 1.0)
    g7 = (gamma - 1.0) / 2.0
    g8 = gamma - 1.0

    if s < uM:
        if pM < pL:
            shL = uL - cL
            if s < shL:
                rho = rhoL
                u = uL
                p = pL
            else:
                cmL = cL * (pM / pL) ** g1
                stL = uM - cmL
                if s > stL:
                    rho = rhoL * (pM / pL) ** (1.0 / gamma)
                    u = uM
                    p = pM
                else:
                    u = g5 * (cL + g7 * uL + s)
                    c = g5 * (cL + g7 * (uL - s))
                    rho = rhoL * (c / cL) ** g4
                    p = pL * (c / cL) ** g3
        else:
            pmL = pM / pL
            sL = uL - cL * np.sqrt(g2 * pmL + g1)
            if s < sL:
                rho = rhoL
                u = uL
                p = pL
            else:
                rho = rhoL * (pmL + g6) / (pmL * g6 + 1.0)
                u = uM
                p = pM
    else:
        if pM > pR:
            pmR = pM / pR
            sR = uR + cR * np.sqrt(g2 * pmR + g1)
            if s > sR:
                rho = rhoR
                u = uR
                p = pR
            else:
                rho = rhoR * (pmR + g6) / (pmR * g6 + 1.0)
                u = uM
                p = pM
        else:
            shR = uR + cR
            if s > shR:
                rho = rhoR
                u = uR
                p = pR
            else:
                cmR = cR * (pM / pR) ** g1
                stR = uM + cmR
                if s < stR:
                    rho = rhoR * (pM / pR) ** (1.0 / gamma)
                    u = uM
                    p = pM
                else:
                    u = g5 * (-cR + g7 * uR + s)
                    c = g5 * (cR - g7 * (uR - s))
                    rho = rhoR * (c / cR) ** g4
                    p = pR * (c / cR) ** g3

    e = p / rho / g8
    return p, u, rho, e


def solve(test, pts):
    uL = test.uL
    uR = test.uR
    cL, cR = test.cL, test.cR
    gamma = test.gamma
    x0 = test.x0

    g4 = 2.0 / (gamma - 1.0)
    assert g4 * (cL + cR) > (uR - uL)

    pM, uM = pStar(test)

    res = {
        "x": np.empty(pts.shape),
        "p": np.empty(pts.shape),
        "v": np.empty(pts.shape),
        "rho": np.empty(pts.shape),
        "energy": np.empty(pts.shape),
    }
    for i, x in enumerate(pts):
        s = (x - x0) / test.t
        p, v, rho, e = sample(test, pM, uM, s)
        res["x"][i] = x
        res["p"][i] = p
        res["v"][i] = v
        res["rho"][i] = rho
        res["energy"][i] = e

    return res


class state:
    def __init__(self, test, gamma=1.4):
        self.name = str(test)

        if test == 0:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.5
            self.t = 0.2
            self.dt = 1e-4

        if test == 1:
            self.rhoL = 1.0
            self.uL = 0.75
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.3
            self.t = 0.2
            self.dt = 1e-4

        if test == 10:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1.0

            self.rhoR = 0.125
            self.uR = 0.0
            self.pR = 0.1

            self.x0 = 0.3
            self.t = 0.2
            self.dt = 1e-4

        if test == 2:
            self.rhoL = 1.0
            self.uL = -2.0
            self.pL = 0.4

            self.rhoR = 1.0
            self.uR = 2.0
            self.pR = 0.4

            self.x0 = 0.5
            self.t = 0.15
            self.dt = 1e-4

        if test == 3:
            self.rhoL = 1.0
            self.uL = 0.0
            self.pL = 1000.0

            self.rhoR = 1.0
            self.uR = 0.0
            self.pR = 0.01

            self.x0 = 0.5
            self.t = 0.012
            self.dt = 5e-6

        if test == 4:
            self.rhoL = 5.99924
            self.uL = 19.5975
            self.pL = 460.894

            self.rhoR = 5.99242
            self.uR = -6.19633
            self.pR = 46.0950

            self.x0 = 0.4
            self.t = 0.035
            self.dt = 1e-5

        if test == 5:
            self.rhoL = 1.0
            self.uL = -19.5975
            self.pL = 1000.0
            self.x0 = 0.8

            self.rhoR = 1.0
            self.uR = -19.5975
            self.pR = 0.01

            self.t = 0.012
            self.dt = 1e-5

        self.cL = np.sqrt(gamma * self.pL / self.rhoL)
        self.cR = np.sqrt(gamma * self.pR / self.rhoR)
        self.gamma = gamma


def plotres(frres, anres, fname=None):

    marker = "o"
    x = frres["x"]
    for el in range(x.shape[-1]):
        plt.plot(
            x[:, el],
            frres["rho"][:, el].ravel(order="F"),
            c="b",
            label=r"$\rho$" if el == 0 else "",
            marker=marker,
        )
        plt.plot(
            x[:, el],
            frres["p"][:, el],
            label=r"$p$" if el == 0 else "",
            marker=marker,
            c="orange",
        )
        plt.plot(
            x[:, el],
            frres["v"][:, el],
            label=r"$v$" if el == 0 else "",
            marker=marker,
            c="g",
        )

    plt.plot(anres["x"], anres["rho"], c="k")
    plt.plot(anres["x"], anres["p"], c="k")
    plt.plot(anres["x"], anres["v"], c="k")

    plt.legend()
    plt.title = fname
    if not fname:
        plt.show()
    else:
        plt.savefig(fname+".png")
    plt.clf()
    plt.close()


if __name__ == "__main__":

    config = {
        "p": 2,
        "quad": "gauss-legendre-lobatto",
        "intg": "rk3",
        "intflux": "hllc",
        "gamma": 1.4,
        "bc": "wall",
        "mesh": "mesh-50.npy",
        "efilt": True,
        "effunc": "numerical",
        "efniter": 2,
    }

    testnum = 0
    # rhospace = momspace = Espace =  np.logspace(-0.8,0.6,12)
    rhospace  = [1.0]
    momspace = [1.0]
    Espace = [1.0]
    norm = "inf"
    savefig = False

    test = state(testnum)
    config["dt"] = test.dt
    config["tend"] = test.t
    config["outfname"] = f"test_{testnum}"
    config["nout"] = 0  # round(test.t / test.dt)

    error = dict()
    normdict = {"inf": np.inf, "L2": 2}
    for frho in rhospace:
        for fmom in momspace:
            for fE in Espace:

                print(f"Working on {frho}, {fmom}, {fE}")

                config["efrhopow"] = frho
                config["efmompow"] = fmom
                config["efEpow"] = fE

                # create system
                a = system(config)

                x = a.x
                rho = np.where(x <= test.x0, test.rhoL, test.rhoR)
                v = np.where(x <= test.x0, test.uL, test.uR)
                p = np.where(x <= test.x0, test.pL, test.pR)

                a.set_ics([rho, v, p])
                try:
                    a.run()
                    fname = ""
                    # Flux Reconstruction results
                    frres = dict()
                    frres["x"] = x
                    frres["rho"] = a.u0[0]
                    frres["v"] = a.u0[1] / frres["rho"]
                    frres["p"] = (config["gamma"] - 1.0) * (a.u0[2] - 0.5 * frres["rho"] * frres["v"] ** 2)

                    # Analyrical Results
                    anres = solve(test, np.ravel(x, order="F"))
                    anres["x"] = a.x.ravel(order="F")

                    for key in ["rho", "v", "p"]:
                        error[key] = np.linalg.norm(frres[key].ravel(order="F") - anres[key], normdict[norm])
                except Exception as e:
                    import sys
                    import traceback

                    print(f"{e}")
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)
                    print("NaN Detected")
                    fname = "NAN-"
                    test.t = a.t
                    for key in ["rho", "v", "p"]:
                        error[key] = "NAN"

                quad = "".join([i[0] for i in a.config["quad"].split("-")])
                fname += f"result{norm}_rhof-{frho:.2f}_momf-{fmom:.2f}_Ef-{fE:.2f}_test-{testnum}_quad-{quad}_neles-{a.neles}_p-{a.order}_efniter-{a.config["efniter"]}"

                # if not Path(fname+".txt").is_file():
                #     with open(f"{fname}.txt", "w") as f:
                #         f.write("frho, fmom, fE, erho, ev, ep\n")

                # if fname.startswith("NAN"):
                #     with open(f"{fname}.txt", "a") as f:
                #         f.write(f"{frho}, {fmom}, {fE}, {error["rho"]}, {error["v"]}, {error["p"]}\n")
                # else:
                #     with open(f"{fname}.txt", "a") as f:
                #         f.write(f"{frho}, {fmom}, {fE}, {error["rho"]}, {error["v"]}, {error["p"]}\n")
                fname = fname.replace(norm, f"_rhof-{frho:.2f}_momf-{fmom:.2f}_Ef-{fE:.2f}")
                plotres(frres, anres, fname if savefig else None)

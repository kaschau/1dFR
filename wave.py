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
import sys
import numpy as np
from oneDFR.elements import system
import matplotlib.pyplot as plt

np.seterr(invalid="raise")


def plotres(frres, fname=None):

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

    plt.legend()
    plt.title = fname
    if not fname:
        plt.show()
    else:
        plt.savefig(fname + ".png")


if __name__ == "__main__":

    config = {
        "intg": "rk4",
        "intflux": "rusanov",
        "gamma": 1.4,
        "bc": "periodic",
    }

    config["mesh"] = "mesh-50.npy"
    config["effunc"] = "nondim"
    config["quad"] = "gauss-legendre"
    config["efilt"] = "linearise"
    config["p"] = 3
    config["e_tol"] = 0

    plot = True
    savefig = False

    config["tend"] = 1.0
    config["nout"] = 0  # round(test.t / test.dt)

    # create system
    a = system(config)

    x = a.x
    rho = 2.0 + np.sin(2 * np.pi * x)
    v = 1.0
    p = 1.0

    # compute CFL = 0.1
    CFL = 0.1
    dx = 1.0 / a.neles / (config["p"] + 1)
    gamma = a.config["gamma"]
    c = np.sqrt(gamma * np.max(p) / np.min(rho)) + np.max(np.abs(v))
    dt = CFL * dx / c
    config["dt"] = dt

    a.set_ics([rho, v, p])
    a.run()

    # Flux Reconstruction results
    frres = dict()
    frres["x"] = x
    frres["rho"] = a.u0[0]
    frres["v"] = a.u0[1] / frres["rho"]
    frres["p"] = (config["gamma"] - 1.0) * (
        a.u0[2] - 0.5 * frres["rho"] * frres["v"] ** 2
    )
    exact = 2.0 + np.sin(2 * np.pi * (x - a.t))

    quad = "".join([i[0] for i in a.config["quad"].split("-")])
    fname = f"quad-{quad}_neles-{a.neles}_p-{a.order}_func-{a.config["effunc"]}"

    if config["efilt"] == "bisect":
        fname += f"_efniter-{a.config["efniter"]}"
    else:
        fname += "_" + str(config["efilt"])

    if plot:
        plotres(frres, fname if savefig else None)
# type: ignore

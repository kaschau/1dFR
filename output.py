import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(
    "/Users/kschau/Dropbox/machines/config/matplotlib/stylelib/whitePresentation.mplstyle"
)


def plot(system, fname=None):
    x = system.x
    u = system.u0
    rho = u[0]
    v = u[1] / rho
    rhoE = u[2]

    p = (system.config["gamma"] - 1.0) * (rhoE - 0.5 * rho * v**2)
    rhos = rho * system.entropy(u)

    marker = "o"
    for el in range(system.neles):
        plt.plot(
            x[:, el],
            u[0, :, el].ravel(order="F"),
            c="b",
            label=r"$\rho$" if el == 0 else "",
            marker=marker,
        )
        plt.plot(
            x[:, el],
            p[:, el],
            label=r"$p$" if el == 0 else "",
            marker=marker,
            c="orange",
        )
        plt.plot(
            x[:, el], v[:, el], label=r"$v$" if el == 0 else "", marker=marker, c="g"
        )
        plt.plot(
            x[:, el],
            rhos[:, el],
            label=r"$\rho s$" if el == 0 else "",
            marker=marker,
            c="r",
        )
    plt.legend(loc="upper right")
    plt.ylim([-0.2, 1.2])
    quad = "".join([i[0] for i in system.config["quad"].split("-")])
    plt.title(f"rule = {quad}, neles = {system.neles}, $p={system.order}$, efniter={system.config["efniter"]}")
    if fname:
        plt.savefig(fname)
        plt.clf()
        plt.close()
    else:
        plt.show()
        plt.close()

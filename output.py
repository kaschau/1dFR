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
    e = system.entropy(u)

    for el in range(system.neles):
        plt.plot(
            x[:, el],
            u[0, :, el].ravel(order="F"),
            c="b",
            label="rho" if el == 0 else "",
            marker="o",
        )
        plt.plot(
            x[:, el], p[:, el], label="p" if el == 0 else "", marker="o", c="orange"
        )
        plt.plot(x[:, el], v[:, el], label="v" if el == 0 else "", marker="o", c="g")
        plt.plot(x[:, el], e[:, el], label="e" if el == 0 else "", marker="*", c="r")
    plt.legend(loc="upper right")
    plt.ylim([-0.2, 1.2])
    if fname:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

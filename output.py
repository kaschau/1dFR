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

    for e in range(system.neles):
        plt.plot(
            x[:, e],
            u[0, :, e].ravel(order="F"),
            c="b",
            label="rho" if e == 0 else "",
            marker="o",
        )
        plt.plot(
            x[:, e], p[:, e], label="p" if e == 0 else "", marker="o", c="orange"
        )
        plt.plot(x[:, e], v[:, e], label="v" if e == 0 else "", marker="o", c="g")
    plt.legend(loc="upper right")
    if fname:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()

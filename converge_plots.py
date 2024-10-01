import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.style.use(
    "/Users/kschau/Dropbox/machines/config/matplotlib/stylelib/whitePresentation.mplstyle"
)

td = "./quadrant/numerical/reg"

files = [
    i for i in os.listdir(td) if i.startswith("converge_") and i.endswith("-20.png")
]

res = {
    1: {15: None, 25: None, 50: None},
    2: {15: None, 25: None, 50: None},
    3: {15: None, 25: None, 50: None},
}

for f in files:
    split = f.split("_")
    neles = int(split[1][1::])
    order = int(split[4][-1])
    error = float(split[-2].split("-")[2])

    res[order][neles] = error

cs = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
for p in res.keys():
    h = np.array([1 / h for h in res[p]])
    e = np.array([res[p][i] for i in res[p]])

    log_h = np.log(h)
    log_e = np.log(e)
    slope, intercept, _, _, _ = linregress(log_h, log_e)
    C = np.exp(intercept)
    C = np.mean(e / h**p)
    ideal = C * h**p
    color = next(cs)
    plt.loglog(h, e, "--", c=color, label=f"$p={p}$")
    plt.loglog(h, ideal, "-", c=color)

plt.xlabel("$h$")
plt.ylabel(r"$||u-u_{exact}||L^{2}$")
plt.legend()
plt.show()

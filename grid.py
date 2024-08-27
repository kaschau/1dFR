import numpy as np

def make_grid(neles, fout):
    eles = np.zeros((neles, 2))
    eles[:, 0] = np.linspace(0, 1, neles + 1)[0:-1]
    eles[:, 1] = np.linspace(0, 1, neles + 1)[1::]

    with open(fout, 'wb') as f:
        np.save(f, eles)

if __name__ == "__main__":
    fout = "mesh.npy"
    neles = 11
    eles = make_grid(neles, fout)
import numpy as np
from numpy.polynomial.legendre import legder as dL
from numpy.polynomial.legendre import legvander
from math import factorial as fac


def binom(n, k):
    return fac(n) / (fac(k) * fac(n - k))


class BasePoly:
    @staticmethod
    def evaluate(y, vdm, a):
        y[:] = np.einsum("xp...,vp...->vx...", vdm, a)

    @staticmethod
    def compute_coeff(a, y, invdm):
        a[:] = np.einsum("xp...,vp...->vx...", invdm, y)

    @staticmethod
    def diff_coeff(a):
        return dL(a, axis=1)


class LegendrePoly(BasePoly):
    def __init__(self, deg):
        self.deg = deg

    def vandermonde(self, x):
        return legvander(x, self.deg)

    def basis_at(self, x):
        deg = self.deg
        return sum(
            [
                binom(deg, k) * binom(deg + k, k) * (0.5 * (x - 1)) ** k
                for k in range(deg + 1)
            ]
        )
        # import numpy as np
        # from numpy.polynomial.legendre import Legendre as L
        # c = np.zeros(self.deg + 1)
        # c[-1] = 1.0
        # return L(c)(x)

    def dbasis_at(self, x):
        deg = self.deg
        if deg > 0:
            return sum(
                [
                    binom(deg, k) * binom(deg + k, k) * 0.5**k * k * (x - 1) ** (k - 1)
                    for k in range(1, deg + 1)
                ]
            )
        else:
            return np.array([2 for _ in range(len(x))])

        # c = np.zeros(deg + 1)
        # c[-1] = 1.0
        # return L(dL(c))(x)

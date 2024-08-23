import numpy as np
from numpy.polynomial.legendre import Legendre as L
from numpy.polynomial.legendre import legder as Lp
from numpy.polynomial.legendre import legvander
from math import factorial as fac


def binom(n, k):
    return fac(n) / (fac(k) * fac(n - k))


class BasePoly:
    @staticmethod
    def evaluate(bank, vdm, a):
        bank[:] = np.einsum("ji...,ki...->kj...", vdm, a)

    @staticmethod
    def compute_coeff(a, y, invdm):
        a[:] = np.einsum("ji...,ki...->kj...", invdm, y)

    @staticmethod
    def diff_coeff(a):
        return Lp(a, axis=1)



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
        # return sum(
        #     [
        #         binom(deg, k) * binom(deg + k, k) * 0.5**k * k * (x - 1) ** (k - 1)
        #         for k in range(deg + 1)
        #     ]
        # )
        c = np.zeros(deg + 1)
        c[-1] = 1.0
        return L(Lp(c))(x)

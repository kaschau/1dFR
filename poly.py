from math import factorial as fac

def binom(n, k):
    return fac(n)/(fac(k)*fac(n-k))

class BasePoly:
    pass

class LegendrePoly(BasePoly):

    def __init__(self, deg):
        self.deg = deg

    def basis_at(self, x):
        deg = self.deg
        return sum([binom(deg, k) * binom(deg + k, k) * (0.5 * (x - 1))**k for k in range(deg+1)])

    def dbasis_at(self, x):
        deg = self.deg
        return sum([binom(deg, k) * binom(deg + k, k) * 0.5 * k * (0.5 * (x - 1))**(k - 1) for k in range(deg+1)])
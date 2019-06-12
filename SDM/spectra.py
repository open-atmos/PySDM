from scipy.stats import lognorm
from scipy.stats import expon
import math


class Exponential:
    def __init__(self, n_part, m_mode, s_geom):  # TODO change name of params?
        self.loc = m_mode
        self.scale = s_geom
        self.n_part = n_part

    def size_distribution(self, m):
        return self.n_part * expon.pdf(m, self.loc, self.scale)

    def stats(self, moments):
        raise expon.stats(loc=self.loc, scale=self.scale, moments=moments)

    def cumulative(self, m):
        return self.n_part * expon.cdf(m, self.loc, self.scale)


class Lognormal:
    def __init__(self, n_part, m_mode, s_geom):
        self.s = math.log(s_geom)
        self.loc = 0
        self.scale = m_mode
        self.n_part = n_part

    def size_distribution(self, m):
        return self.n_part * lognorm.pdf(m, self.s, self.loc, self.scale)

    def stats(self, moments):
        return lognorm.stats(self.s, loc=self.loc, scale=self.scale, moments=moments)

    def cumulative(self, m):
        result = self.n_part * lognorm.cdf(m, self.s, self.loc, self.scale)
        return result

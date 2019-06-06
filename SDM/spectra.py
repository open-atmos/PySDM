from scipy.stats import lognorm
import math

class Lognormal():
	def __init__(self, n_part, m_mode, s_geom):
		self.s = math.log(s_geom) 
		self.loc = 0
		self.scale = m_mode
		self.n_part = n_part

	def size_distribution(self, m):
		return self.n_part * lognorm.pdf(m, self.s, self.loc, self.scale)

	def stats(self, moments):
		return lognorm.stats(self.s, loc=self.loc, scale=self.scale, moments=moments)

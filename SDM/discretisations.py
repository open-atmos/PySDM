import numpy as np


# TODO Discretisation.linear(function, bins, (min, max))
class Linear:
	def __init__(self, m_min, m_max):
		self.m_min = m_min
		self.m_max = m_max
		
	def __call__(self, n_sd, spectrum):
		m, dr = np.linspace(self.m_min, self.m_max, num=n_sd, retstep=True)
		n = dr * spectrum.size_distribution(m)
		return m, n.round().astype(int)

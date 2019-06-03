import numpy as np

class State:
	def __init__(self, n_particles, discretise, spectrum):
		self.r, self.n = discretise(n_particles, spectrum)

	def moment(self, k):
		return np.average(self.r**k, weights=self.n)

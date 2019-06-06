import numpy as np


class State:
	def __init__(self, m, n):  # n_particles, discretise, spectrum):
		assert m.shape == n.shape
		assert len(m.shape) == 1
		self.m = m.copy()
		self.n = n.copy()

	def moment(self, k):
		return np.average(self.m ** k, weights=self.n)

	def collide(self, i, j, gamma):
		if self.n[i] < self.n[j]:
			i, j = j, i

		gamma = min(gamma, self.n[i] // self.n[j])

		if self.n[j] != 0:
			self.n[i] -= gamma * self.n[j]
			self.m[j] += gamma * self.m[i]

	def __len__(self):
		return self.m.shape[0]

	def __getitem__(self, item):
		return self.m[item], self.n[item]
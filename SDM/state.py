import numpy as np


class State:
	# TODO: rethink if m and n should not be in one array
	def __init__(self, m, n):
		assert m.shape == n.shape
		assert len(m.shape) == 1
		self.m = m.copy()
		self.n = n.copy()

	def _sort(self, key):
		idx = np.argsort(key)
		self.n = self.n[idx]
		self.m = self.m[idx]

	def sort_by_m(self):
		self._sort(self.m)

	def sort_by_n(self):
		self._sort(self.n)

	def unsort(self):
		idx = np.random.permutation(np.arange(len(self)))
		self.m = self.m[idx]
		self.n = self.n[idx]

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
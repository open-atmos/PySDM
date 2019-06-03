import numpy as np

class Linear:
	def __init__(self, r_min, r_max):
		self.r_min = r_min
		self.r_max = r_max
		
	def __call__(self, n_sd, spectrum):
		r, dr = np.linspace(self.r_min, self.r_max, num=n_sd, retstep=True)
		n = dr * spectrum.size_distribution(r)
		print(n)
		return r, n.round().astype(int) 

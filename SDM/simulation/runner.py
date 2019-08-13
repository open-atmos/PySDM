"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.simulation.stats import Stats


# TODO inject callback functions
class Runner:
	def __init__(self, state, dynamics):
		self.state = state
		self.dynamics = dynamics
		self.n_steps = 0
		self.stats = Stats()  # TODO: inject?

	def run(self, steps):
		with self.stats:
			for _ in range(steps):
				for d in self.dynamics:
					d(self.state)
		self.n_steps += steps

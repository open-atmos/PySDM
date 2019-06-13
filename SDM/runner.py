"""
Created at 03.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Runner:
	def __init__(self, state, dynamics):
		self.state = state
		self.dynamics = dynamics

	def run(self, nt):
		for _ in range(nt):
			for d in self.dynamics:
				d(self.state)

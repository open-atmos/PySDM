class Runner:
	def __init__(self, state, dynamics):
		self.state = state
		self.dynamics = dynamics

	def run(self, nt):
		for _ in range(nt):
			for d in self.dynamics:
				d.dynamics.step(self.state)

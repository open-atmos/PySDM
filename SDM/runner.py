class Runner:
	def __init__(self, state, dynamics):
		self.state = state
		self.dynamics = dynamics

	def run(self, nt):
		for _ in range(nt):
			self.dynamics.step(self.state)

class Runner:
	def __init__(state, dynamics):
		self.state = state
		self.dynamics = dynamics

	def run(nt):
		for _ in range(nt):
			self.dynamics.step(self.state)

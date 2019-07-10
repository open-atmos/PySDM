from matplotlib import pyplot
from SDM.state import State


def plot(state):
	state.sort_by_m()

	pyplot.plot(state.x, state.n)
	pyplot.grid()
	pyplot.show()


def plot_mn(m, n):
	state = State(m, n)
	plot(state)

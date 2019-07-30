from matplotlib import pyplot
from SDM.state import State


def plot(state):
	state.sort_by('x')

	pyplot.plot(state['x'], state['n'])
	pyplot.grid()
	pyplot.show()


def plot_mn(x, n):
	state = State({'x': x, 'n': n})
	plot(state)

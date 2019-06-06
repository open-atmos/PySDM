from matplotlib import pyplot


def plot(state):
	pyplot.plot(state.m, state.n)
	pyplot.grid()
	pyplot.show()	

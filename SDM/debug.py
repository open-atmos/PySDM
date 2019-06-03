from matplotlib import pyplot

def plot(state):
	pyplot.plot(state.r, state.n)
	pyplot.grid()
	pyplot.show()	

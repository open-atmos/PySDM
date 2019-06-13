import numpy as np


def __discritise(grid, spectrum):
    m = grid[1: -1: 2]
    cdf = spectrum.cumulative(grid[0::2])
    n = cdf[1:] - cdf[0:-1]
    return m, n.round().astype(int)


def linear(n_sd, spectrum, range):
    assert range[0] >= 0
    assert range[1] > range[0]

    grid = np.linspace(range[0], range[1], num=2 * n_sd + 1)

    return __discritise(grid, spectrum)


def logarithmic(n_sd, spectrum, range):
    assert range[0] > 0
    assert range[1] > range[0]

    start = np.log10(range[0])
    stop = np.log10(range[1])

    grid = np.logspace(start, stop, num=2 * n_sd + 1)

    return __discritise(grid, spectrum)

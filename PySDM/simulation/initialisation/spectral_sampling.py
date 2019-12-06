import numpy as np


def __sample(grid, spectrum):
    x = grid[1: -1: 2]
    cdf = spectrum.cumulative(grid[0::2])
    y_float = cdf[1:] - cdf[0:-1]

    return x, y_float


def linear(n_sd, spectrum, range):
    assert range[0] >= 0
    assert range[1] > range[0]

    grid = np.linspace(range[0], range[1], num=2 * n_sd + 1)

    return __sample(grid, spectrum)


def logarithmic(n_sd, spectrum, range):
    assert range[0] > 0
    assert range[1] > range[0]

    start = np.log10(range[0])
    stop = np.log10(range[1])

    grid = np.logspace(start, stop, num=2 * n_sd + 1)

    return __sample(grid, spectrum)


def constant_multiplicity(n_sd, spectrum, range):
    assert range[0] > 0
    assert range[1] > range[0]

    cdf_min = spectrum.cumulative(range[0])
    cdf_max = spectrum.cumulative(range[1])

    cdfarg = np.linspace(cdf_min, cdf_max, num=2 * n_sd + 1)
    cdfarg /= spectrum.norm_factor
    grid = spectrum.percentiles(cdfarg)

    assert np.isfinite(grid).all()

    return __sample(grid, spectrum)


# TODO: sample randomly

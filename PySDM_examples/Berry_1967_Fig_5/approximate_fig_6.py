"""
Created at 06.06.2020
"""

from PySDM.physics import constants as const
from PySDM.dynamics.coalescence.kernels.gravitational import __linear_collection_efficiency
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

um = const.si.um


def print_collection_efficiency_portrait(D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg):
    size = 2*5.236
    plt.figure(num=1, figsize=(size / 2, size))
    radii = (8, 10, 14, 16, 20, 30, 40, 60, 80, 136)
    radii = (r * const.si.um for r in radii)
    points = 200
    Y_c = np.zeros(points)
    p = np.linspace(0, 1, points)
    for r in radii:
        for i in range(len(p)):
            Y_c[i] = __linear_collection_efficiency(r, p[i]*r, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg)
        plt.plot(p, Y_c, label=f'{r/const.si.um}um')
    # plt.legend()
    xticks = np.arange(11)/10
    yticks = np.arange(21)/10
    plt.xticks(xticks, xticks)
    plt.yticks(yticks, yticks)
    plt.grid()
    plt.show()


p = np.array((.2, .4, .6, .8, .9, 1.))
expected = [[0, 0, 8.3, 0, 0, 0],  # 8
            [0, 36, 55.6, 24, 0, 0],  # 10
            [27, 85, 105.3, 96.2, 49.6, 0],  # 14
            [46.6, 94.7, 116.5, 107.5, 84.2, 0],  # 16
            [70, 108.2, 130.8, 136, 112.8, 0], #  20
            [117, 138, 158, 179, 189, 0]  # 136
            ]
expected = np.abs(expected)
expected /= 100
radii = np.array((8*um, 10*um, 14*um, 16*um, 20*um, 136*um))


def error(x):
    D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = x
    Y_c = np.zeros(expected.shape)
    for i in range(len(radii)):
        for j in range(len(p)):
            Y_c[i, j] = __linear_collection_efficiency(radii[i], p[j] * radii[i], D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg)
    return np.sum((Y_c - expected)**2) + 0 * np.abs(Y_c[0, 2] - expected[0, 2])


def error2(x):
    D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = x
    Y_c = np.zeros(expected.shape)
    for i in range(len(radii)):
        for j in range(len(p)):
            Y_c[i, j] = __linear_collection_efficiency(radii[i], p[j] * radii[i], D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg)
    return np.sum((Y_c[:-1] - expected[:-1])**2) + 1000 * np.abs(Y_c[0, 2] - expected[0, 2])


x = [-7, 1.78, -20.5, 1.73, .26, 1.47, 1, .82, -0.003, 4.4, 8]
A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = x


if __name__ == '__main__':
    x0 = np.array([-27, 1.65, -58, 1.9, 1, 1.13, 1, 1, 0.004, 4, 8])
    x = np.array([-7, 1.78, -20.5, 1.73, .26, 1.47, 1, .82, -0.003, 4.4, 8])
    print_collection_efficiency_portrait(*x)
    res = minimize(error, x)
    print(res.x)
    print(error(res.x))
    print_collection_efficiency_portrait(*res.x)
    res = minimize(error2, res.x)
    print(res.x)
    print(error(x), error(res.x), error(res.x))
    print_collection_efficiency_portrait(*res.x)



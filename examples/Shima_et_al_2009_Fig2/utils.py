"""
Created at 09.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from numpy import pi

kg2g = 1e3
m2um = 1e6


def x2r(x):
    return (x * 3/4 / pi)**(1/3)


def r2x(r):
    return 4/3 * pi * r**3


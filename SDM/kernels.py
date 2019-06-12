"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Golovin:
    def __init__(self, b):
        self.b = b

    def __call__(self, m1, m2):

        return self.b * (m1 + m2)

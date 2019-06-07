"""
Created at 06.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Dynamic:
    def __init__(self, undertaker, collider):
        self.undertaker = undertaker
        self.collider = collider

    def step(self, state):
        self.collider(state)
        self.undertaker(state)

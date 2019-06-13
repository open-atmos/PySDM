"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Resize:
    def __call__(self, state):
        # TODO dependency state items
        idx_valid = state.n != 0
        state.n = state.n[idx_valid]
        state.m = state.m[idx_valid]


class Recycle:
    def __call__(self, state):
        #TODO: 	state.sort_by_n()

        raise NotImplementedError

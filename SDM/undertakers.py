"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class Resize:
    def __call__(self, state, attr='n'):
        # TODO dependency state items!!!
        idx_valid = state[attr] != 0
        state.data = state.data[:, idx_valid]


class Recycle:
    def __call__(self, state, attr='n'):
        # TODO: state.sort_by(attr)
        raise NotImplementedError

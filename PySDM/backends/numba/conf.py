"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


JIT_FLAGS = {
    "parallel": False,
    "fastmath": True,
    "error_model": 'numpy',  # TODO: 'numpy' would be faster, but nans make condensation loop forever
#    "boundscheck": False
}

# TODO: cache!
# TODO: enforce cache for mpdata...
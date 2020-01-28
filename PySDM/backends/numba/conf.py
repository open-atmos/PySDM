"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


JIT_FLAGS = {
    "parallel": True,
    "fastmath": True,
    "error_model": 'python'  # TODO: 'numpy' would be faster, but nans make condensation loop forever
}
"""
Created at 18.05.21 by edejong
"""

import numpy as np


class SLAMS:

    def __init__(self):
        self.core = None

    def __call__(self, output, is_first_in_pair):
        p = 0.0
        r = np.random.rand()       # TODO: optimize random generator outside the fragmentation function
        nf = 1
        for i in range(22):
            p += 0.91 * (i + 2)**(-1.56)
            if (r < p):
                nf = i + 2
                break
        output *= 0
        output += nf

    def register(self, builder):
        self.core = builder.core

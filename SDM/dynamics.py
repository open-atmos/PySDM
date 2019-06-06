"""
Created at 06.06.2019

@author: Piotr Bartman
"""

import numpy as np


class Dynamic:
    def __init__(self, probability):
        self.probability = probability

    def step(self, state):
        n = len(state)

        # toss pairs
        idx = np.random.permutation(np.arange(n))

        # collide iterating over pairs
        rand = np.random.uniform(0, 1, n//2)
        prob = np.empty_like(rand)

        for i in range(1, n, 2):
            prob[i//2] = self.probability(state[idx[i]], state[idx[i-1]])

        gamma = np.floor(prob) + np.where(rand < prob - np.floor(prob), 1, 0)

        for i in range(1, n, 2):
            state.collide(idx[i], idx[i-1], gamma)

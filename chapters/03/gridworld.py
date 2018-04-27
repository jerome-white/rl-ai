import random
import itertools as it
import collection as cl

import numpy as np

Step = cl.nametuple('Step', 'reward, action')

def navigator():
    for (i, j) in it.permutations(range(-1, 2), 2):
        if not i or not j:
            yield lambda x, y: (x + i, y + j)

class Action:
    def __init__(self):
        self.reward = 0
        self.neighborhood = []

class Grid:
    def __init__(self, rows, columns=None):
        if columns is None:
            columns = rows

        self.width = columns
        self.current = None
        self.grid = [ Action() for _ in range(rows * columns) ]

        for (coordinate, action) in np.ndenumerate(self.grid):
            for f in navigator():
                (x, y) = f(*coordinate)
                inbounds = 0 <= x < rows and 0 <= y < columns
                ptr = self.get(x, y) if inbounds else None
                action.neighborhood.append(ptr)

    def get(self, row, column):
        return self.grid[row * self.width + column]

    def __iter__(self):
        self.current = random.choice(self.grid)
        return self

    def __next__(self):
        action = random.choice(self.current.neighborhood)

        if action is None:
            reward = -1
        else:
            reward = self.current.reward
            self.current = action

        return Step(reward, self.current)

import random
import itertools as it

import numpy as np

def navigator():
    for (i, j) in it.permutations(range(-1, 2), 2):
        if not i or not j:
            yield lambda x, y: (x + i, y + j)

class Action:
    def __init__(self):
        self.reward = 0
        self.neighborhood = []

    def __next__(self):
        return random.choice(self.neighborhood)

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
        action = next(self.current)
        if action is None:
            reward = -1
        else:
            reward = action.reward
            self.current = action

        return (reward, self.current)

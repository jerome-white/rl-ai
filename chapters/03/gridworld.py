import random
import itertools as it

import numpy as np

def navigator():
    for (i, j) in it.permutations(range(-1, 2), 2):
        if not i or not j:
            yield lambda x, y: (x + i, y + j)

class Action:
    def __init__(self, neighbors=None):
        self.estimate = 0
        self.neighbors = [] if neighbors is None else neighbors

    def __float__(self):
        return float(self.estimate)

    def increment(self, estimate):
        self.estimate += estimate

class Transition:
    def __init__(self, action, reward=0, probability=1):
        self.action = action
        self.reward = reward
        self.probability = probability

class Grid:
    def __init__(self, rows, columns=None):
        if columns is None:
            columns = rows

        self.ptr = None
        self.grid = []

        for _ in range(rows):
            self.grid.append([ Action() for _ in range(columns) ])

        for (coordinate, action) in np.ndenumerate(self.grid):
            for f in navigator():
                (x, y) = f(*coordinate)
                if 0 <= x < rows and 0 <= y < columns:
                    trans = Transition(self.grid[x][y])
                else:
                    trans = Transition(None, -1)
                action.neighbors.append(trans)

        for (_, action) in np.ndenumerate(self.grid):
            for i in action.neighbors:
                i.probability = 1 / len(action.neighbors)

    def __str__(self):
        sep = ('+-' + '-' * 6) * len(self.grid[0]) + '+'
        table = [ sep ]

        for row in self.grid:
            line = map('{0:5.2f}'.format, map(float, row))
            table.extend([
                '| ' + ' | '.join(line) + ' |',
                sep
            ])

        return '\n'.join(table)

    # def __next__(self):
    #     off = self.on
    #     transition = random.choice(self.on.neighbors)

    #     if transition.action is None:
    #         reward = -1
    #     else:
    #         reward = transition.reward
    #         self.on = transition.action

    #     return (off, reward, self.on)

    def walk(self):
        for (coordinate, action) in np.ndenumerate(self.grid):
            print(coordinate)
            est = action.estimate
            for transition in action.neighbors:
                if transition.action is None:
                    estimate = est
                else:
                    estimate = transition.action.estimate
                yield (action, transition, estimate)

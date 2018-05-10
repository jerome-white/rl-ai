import os
import time
from argparse import ArgumentParser

import gridworld as gw

class Estimate(list):
    def __init__(self, dimensions):
        (rows, columns) = dimensions
        for m in range(rows):
            self.append([ 0 ] * columns)

    def __str__(self):
        table = []
        sep = ('+' + '-' * 7) * len(self[0]) + '+'

        for row in self:
            line = [ '{0:5.2f}'.format(x) for x in row ]
            table.extend([
                sep,
                '| ' + ' | '.join(line) + ' |',
            ])
        if table:
            table.append(sep)

        return '\n'.join(table)

    def __sub__(self, other):
        diff = 0

        for (i, j) in zip(self, other):
            for (x, y) in zip(i, j):
                diff += x - y

        return diff

    def update(self, state, value):
        self[state.x][state.y] = value

arguments = ArgumentParser()
arguments.add_argument('--discount', type=float)
args = arguments.parse_args()

grid = gw.SpecialGrid()
before = Estimate(grid.dimensions)

while True:
    after = Estimate(grid.dimensions)

    for (state, actions) in grid:
        p = 1 / len(actions)
        estimate = 0
        for a in actions:
            est = before[a.state.x][a.state.y]
            estimate += p * (a.reward + args.discount * est)
        after[state.x][state.y] = estimate

    os.system('clear')
    print(before, before - after, sep='\n')
    time.sleep(0.5)

    before = after

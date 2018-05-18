import os
import time
from argparse import ArgumentParser

import gridworld as gw
from estimate import Estimate

arguments = ArgumentParser()
arguments.add_argument('--discount', type=float)
arguments.add_argument('--speed', type=float, default=0.5)
args = arguments.parse_args()

grid = gw.SpecialGrid()
before = Estimate(grid.dimensions)

while True:
    after = Estimate(grid.dimensions)

    for (state, actions) in grid:
        estimate = before.estimates(actions, args.discount)
        after[state.x][state.y] = max(estimate)

    os.system('clear')
    print(before, before - after, sep='\n')
    time.sleep(args.speed)

    before = after

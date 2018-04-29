import os
import time
import itertools as it
from argparse import ArgumentParser

import numpy as np

from gridworld import Grid, Action, Transition

arguments = ArgumentParser()
arguments.add_argument('--discount', type=float)
args = arguments.parse_args()

grid = Grid(5)
grid.grid[0][1] = Action([ Transition(grid.grid[4][1], 10) ])
grid.grid[0][3] = Action([ Transition(grid.grid[2][3], 5) ])

for i in it.count():
    diff = 0
    for (ac, tr, rw) in grid.walk():
        estimate = tr.probability * (tr.reward + args.discount * rw)
        diff += abs(estimate - ac.estimate)
        ac.estimate += estimate
        print(tr.probability,'* (',tr.reward,'+',args.discount,'*',rw,') :',
              estimate, ac.estimate)
    print(grid, i, diff, sep='\n')
    # time.sleep(0.5)    
    # os.system('clear')
    if i > 3:
        break

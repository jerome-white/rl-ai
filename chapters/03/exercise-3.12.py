import os
import time
from argparse import ArgumentParser

from estimate import BackupSolver
from gridworld import SpecialGrid

class Solver(BackupSolver):
    def probability(self, actions):
        return 1

    def collect(self, estimates):
        return max(estimates)

arguments = ArgumentParser()
arguments.add_argument('--discount', type=float)
arguments.add_argument('--speed', type=float, default=0.5)
args = arguments.parse_args()

grid = SpecialGrid()
solver = Solver(grid, args.discount)
for (i, (estimate, difference)) in enumerate(solver):
    os.system('clear')
    print('{0}\n{1}: {2}'.format(estimate, i, difference))
    time.sleep(args.speed)

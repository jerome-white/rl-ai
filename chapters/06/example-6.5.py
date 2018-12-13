import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

from gridworld import GridWorld

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--episodes', type=int, default=8000)
args = arguments.parse_args()

Q = np.zeros((7, 10))

for i in range(args.episodes):
    grid = GridWorld(*Q.shape)

    steps = 0
    s = grid.start
    a = policy.select(s, Q)
    while s != grid.goal:
        (s_, r) = grid.walk(s, a)
        a_ = policy.select(s_, Q)
        Q[(s, a)] += self.alpha * (r + self.gamma * Q[(s_, a_)] - Q[(s, a)])
        (s, a) = (s_, a_)
        steps += 1

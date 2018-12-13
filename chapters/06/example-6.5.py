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
start = (3, 0)
goal = (3, 8)

for i in range(args.episodes):
    grid = GridWorld(*map(State, (start, goal, Q.shape)))

    steps = 0
    state = grid.state
    action = policy.select(state, Q)
    while grid:
        (state_, reward) = grid.walk(action)
        action = policy.select(state_, Q)

        Q[state] += self.alpha * (r + self.gamma * Q[state_] - Q[state])

        steps += 1

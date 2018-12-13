import sys
import csv
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

from gridworld import State, Grid, EpsilonGreedyPolicy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--episodes', type=int, default=8000)
args = arguments.parse_args()

Q = np.zeros((7, 10))
start = State(3, 0)
goal = State(3, 8)

fieldnames = [ 'episodes', 'steps' ]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for episode in range(args.episodes):
    logging.info(episode)

    grid = Grid(Q.shape, goal)
    policy = EpsilonGreedyPolicy(grid.shape, args.epsilon)

    steps = 0
    state = start
    action = policy.choose(state, Q)

    while state != goal:
        logging.debug('s: {0}, a: {1}'.format(state, action))

        (state_, reward) = grid.walk(state, action)
        action = policy.choose(state_, Q)

        Q[state] += args.alpha * (reward + args.gamma * Q[state_] - Q[state])

        state = state_
        steps += 1

    writer.writerow(dict(zip(fieldnames, (episode, steps))))

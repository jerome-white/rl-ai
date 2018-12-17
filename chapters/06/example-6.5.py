import sys
import csv
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

from gridworld import State, WindyGrid, EpsilonGreedyPolicy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--episodes', type=int, default=8000)
arguments.add_argument('--rows', type=int, default=7)
arguments.add_argument('--columns', type=int, default=10)
args = arguments.parse_args()

start = State(3, 0)
goal = State(3, 8)

fieldnames = [ 'episodes', 'steps' ]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for episode in range(args.episodes):
    logging.info(episode)

    grid = WindyGrid(args.rows, args.columns, goal)
    Q = EpsilonGreedyPolicy(grid, args.epsilon)

    steps = 0
    state = start
    action = Q.select(state)

    while state != goal:
        (state_, reward) = grid.navigate(state, action)
        action_ = Q.select(state_)

        now = (state, action)
        later = (state_, action_)

        Q[now] += args.alpha * (reward + args.gamma * Q[later] - Q[now])
        # logging.debug("s: {}, a: {}, r: {}, s': {}, Q: {}"
        #               .format(state, action, reward, state_, Q[state]))

        (state, action) = (state_, action_)
        steps += 1

    writer.writerow(dict(zip(fieldnames, (episode, steps))))

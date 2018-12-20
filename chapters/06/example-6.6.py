import sys
import csv
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

import gridworld as gw
from cliff import Cliff

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--episodes', type=int, default=8000)
arguments.add_argument('--rows', type=int, default=4)
arguments.add_argument('--columns', type=int, default=12)
args = arguments.parse_args()

start = gw.State(3, 0)
goal = gw.State(3, args.columns - 1)

grid = Cliff(args.rows, args.columns, goal, start)
Q = gw.EpsilonGreedyPolicy(grid, args.epsilon)

fieldnames = [ 'episodes', 'steps' ]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
writer.writeheader()

for episode in range(args.episodes):
    logging.info(episode)

    steps = 0
    state = start

    while state != goal:
        action = Q.select(state)        
        (state_, reward) = grid.navigate(state, action)

        Q[now] += args.alpha * (reward + args.gamma * Q.amax(state_) - Q[now])
        logging.debug("s: {}, a: {}, r: {}, s': {}, Q: {}"
                      .format(state, action, reward, state_, Q[now]))

        state = state_
        steps += 1

    writer.writerow(dict(zip(fieldnames, (episode, steps))))

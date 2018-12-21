import sys
import csv
import logging
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np

import gridworld as gw

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def run(grid, args):
    steps = 0
    start = gw.State(3, 0)

    Q = gw.EpsilonGreedyPolicy(grid, args.epsilon)

    while steps < args.time_steps:
        state = start
        action = Q.select(state)

        while state != grid.goal:
            (state_, reward) = grid.navigate(state, action)
            action_ = Q.select(state_)

            now = (state, action)
            later = (state_, action_)

            Q[now] += args.alpha * (reward + args.gamma * Q[later] - Q[now])
            logging.debug("s: {}, a: {}, r: {}, s': {}, Q: {}"
                          .format(state, action, reward, state_, Q[now]))

            (state, action) = (state_, action_)

            steps += 1
            if steps > args.time_steps:
                break

        yield steps

def func(incoming, outgoing, args):
    grid = gw.GridWorld((7, 10),
                        gw.State(3, 8),
                        gw.FourPointCompass(),
                        gw.Wind())

    while True:
        order = incoming.get()
        logging.info(order)

        for i in enumerate(run(grid, args)):
            outgoing.put((order, *i))
        outgoing.put(None)

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--time-steps', type=int, default=8000)
arguments.add_argument('--repeat', type=int, default=1)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

with mp.Pool(args.workers, func, (outgoing, incoming, args)):
    jobs = 0
    for i in range(args.repeat):
        outgoing.put(i)
        jobs += 1

    writer = None
    fieldnames = [ 'order', 'episodes', 'steps' ]

    while jobs:
        result = incoming.get()
        if result is None:
            jobs -= 1
        else:
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(dict(zip(fieldnames, result)))

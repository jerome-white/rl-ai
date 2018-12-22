import sys
import csv
import logging
import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser

import gridworld as gw

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def func(incoming, outgoing, args):
    grid = gw.GridWorld((7, 10),
                        gw.State(3, 7),
                        gw.FourPointCompass(),
                        gw.Wind())

    while True:
        order = incoming.get()

        start = gw.State(3, 0)
        policy = gw.EpsilonGreedyPolicy(grid, args.epsilon)
        process = gw.sarsa(grid, start, policy, args.alpha, args.gamma)

        for (i, (episode, *_)) in enumerate(process):
            if i > args.time_steps:
                break
            message = (order, episode, i)
            logging.info('{} {} {}'.format(*message))
            outgoing.put(message)
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
    fieldnames = [ 'order', 'episode', 'step' ]

    while jobs:
        result = incoming.get()
        if result is None:
            jobs -= 1
        else:
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(dict(zip(fieldnames, result)))

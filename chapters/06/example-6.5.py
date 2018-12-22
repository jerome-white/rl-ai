import sys
import csv
import logging
import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import gridworld as gw

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def func(incoming, outgoing, args):
    keys = [ 'order', 'experiment', 'episode', 'step' ]

    dimensions = (7, 10)
    start = gw.State(3, 0)
    goal = gw.State(3, 7)

    while True:
        (order, name, config) = incoming.get()

        logging.info('{} {}'.format(order, name))

        grid = gw.GridWorld(dimensions, goal, *[ x() for x in config ])
        policy = gw.EpsilonGreedyPolicy(grid, args.epsilon)
        process = gw.sarsa(grid, start, policy, args.alpha, args.gamma)

        for (i, (episode, *_)) in enumerate(process):
            if i > args.time_steps:
                break
            result = dict(zip(keys, (order, name, episode, i)))
            outgoing.put(result)
        outgoing.put(None)

def do(args):
    incoming = mp.Queue()
    outgoing = mp.Queue()

    with mp.Pool(args.workers, func, (outgoing, incoming, args)):
        experiments = {
            'example 6.5': (gw.FourPointCompass, gw.Wind),
            'exercise 6.6a': (gw.KingsMoves, gw.Wind),
            'exercise 6.6b': (gw.KingsMovesNinth, gw.Wind),
            'exercise 6.7': (gw.KingsMoves, gw.StochasticWind),
        }

        jobs = 0
        for i in experiments.items():
            for j in range(args.repeat):
                outgoing.put((j, *i))
                jobs += 1

        while jobs:
            result = incoming.get()
            if result is None:
                jobs -= 1
            else:
                yield result

arguments = ArgumentParser()
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--time-steps', type=int, default=8000)
arguments.add_argument('--repeat', type=int, default=1)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

df = pd.DataFrame.from_dict(do(args))
logging.info('plotting {}'.format(len(df)))
sns.lineplot(x='step',
             y='episode',
             hue='experiment',
             data=df)
plt.grid(True)
plt.savefig('figure-6.11.png')

import logging
import functools as ft
import multiprocessing as mp
from argparse import ArgumentParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import gridworld as gw
from cliff import Cliff

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class ClassicGrid(gw.GridWorld):
    def __init__(self, shape, goal):
        super().__init__(shape, goal, gw.FourPointCompass(), gw.Wind())

def func(incoming, outgoing, args):
    keys = [ 'order', 'experiment', 'episode', 'reward' ]

    shape = (4, 12)
    (row, column) = [ x - 1 for x in shape ]
    start = gw.State(row, 0)
    goal = gw.State(row, column)

    while True:
        (order, experiment, Learning) = incoming.get()

        logging.info('{} {}'.format(order, experiment))

        grid = Cliff(shape, goal, start)
        policy = gw.EpsilonGreedyPolicy(grid, args.epsilon)
        process = Learning(grid, start, policy, args.alpha, args.gamma)

        for (episode, _, reward) in process:
            if episode > args.episodes:
                break
            result = dict(zip(keys, (order, experiment, episode, reward)))
            outgoing.put(result)
        outgoing.put(None)

def do(args):
    incoming = mp.Queue()
    outgoing = mp.Queue()

    with mp.Pool(args.workers, func, (outgoing, incoming, args)):
        experiments = {
            'SARSA': gw.Sarsa,
            'Q-learning': gw.QLearning,
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
arguments.add_argument('--episodes', type=int, default=500)
arguments.add_argument('--repeat', type=int, default=1)
arguments.add_argument('--smoothing', type=int)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

df = pd.DataFrame.from_dict(do(args))
df = (df
      .groupby(['order', 'episode', 'experiment'])
      .sum()
      .rolling(args.smoothing)
      .mean()
      .reset_index())

logging.info('plotting {}'.format(len(df)))
sns.lineplot(x='episode',
             y='reward',
             hue='experiment',
             data=df)
plt.ylim((-100, 0))
plt.grid(True)
plt.savefig('figure-6.14.png')

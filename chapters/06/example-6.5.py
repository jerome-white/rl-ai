import logging
import multiprocessing as mp
from argparse import ArgumentParser
from collections import OrderedDict

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
        (order, experiment, config) = incoming.get()

        logging.info('{} {}'.format(order, experiment))

        (compass, wind) = config
        grid = gw.WindyGridWorld(dimensions, goal, compass(), wind())
        policy = gw.EpsilonGreedyPolicy(grid, args.epsilon)
        process = gw.Sarsa(grid, start, policy, args.alpha, args.gamma)

        for (i, (episode, *_)) in enumerate(process):
            if i > args.time_steps:
                break
            result = dict(zip(keys, (order, experiment, episode, i)))
            outgoing.put(result)
        outgoing.put(None)

def do(args, experiments):
    incoming = mp.Queue()
    outgoing = mp.Queue()

    with mp.Pool(args.workers, func, (outgoing, incoming, args)):
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

experiments = OrderedDict([
    ('example 6.5', (gw.FourPointCompass, gw.Wind)),
    ('exercise 6.6a', (gw.KingsMoves, gw.Wind)),
    ('exercise 6.6b', (gw.KingsMovesNinth, gw.Wind)),
    ('exercise 6.7', (gw.KingsMoves, gw.StochasticWind)),
])

df = pd.DataFrame.from_dict(do(args, experiments))

logging.info('plotting {}'.format(len(df)))
sns.lineplot(x='step',
             y='episode',
             hue='experiment',
             hue_order=experiments.keys(),
             data=df)
plt.grid(True)
plt.savefig('figure-6.11.png')

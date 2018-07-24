import math
import logging
import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm

from walk import TemporalDifference, MonteCarlo

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def func(incoming, outgoing, states, episodes, runs):
    exclusion = states + 1
    actual = [ x / exclusion for x in range(1, exclusion) ]

    Model = {
        'td': TemporalDifference,
        'mc': MonteCarlo,
    }

    while True:
        (method, alpha) = incoming.get()

        for i in range(runs):
            logging.info('{0} {1} {2}'.format(method, alpha, i))

            model = Model[method](states, episodes, alpha)
            for (j, m) in enumerate(model):
                predicted = [ m[x] for x in model.states ]
                rmse = math.sqrt(skm.mean_squared_error(actual, predicted))

                outgoing.put({
                    'model': '{0}:{1}'.format(method, alpha),
                    'run': i,
                    'step': j,
                    'RMSE': rmse,
                })

        outgoing.put(None)

def do(args):
    incoming = mp.Queue()
    outgoing = mp.Queue()

    initargs = (outgoing, incoming, args.states, args.episodes, args.runs)

    with mp.Pool(args.workers, func, initargs):
        td_alpha = np.linspace(0.05, 0.15, 3)
        mc_alpha = np.linspace(0.01, 0.04, 4)

        for (model, alpha) in zip(('td', 'mc'), (td_alpha, mc_alpha)):
            for a in alpha:
                outgoing.put((model, a))

        jobs = len(td_alpha) + len(mc_alpha)
        while jobs:
            value = incoming.get()
            if value is None:
                jobs -= 1
            else:
                yield value

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--runs', type=int, default=100)
arguments.add_argument('--episodes', type=int, default=100)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

df = pd.DataFrame(do(args))

ax = sns.pointplot(x='step',
                   y='RMSE',
                   hue='model',
                   ci=None,
                   data=df)
ax.get_figure().savefig('example-6.2b.png')

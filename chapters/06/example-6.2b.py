import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt

from walk import TemporalDifference, MonteCarlo

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

System = cl.namedtuple('System', 'model, alpha')

def func(incoming, outgoing, states, episodes):
    exclusion = states + 1
    actual = [ x / exclusion for x in range(1, exclusion) ]

    Model = {
        'td': TemporalDifference,
        'mc': MonteCarlo,
    }

    while True:
        system = incoming.get()
        logging.debug(system)

        rmse = []
        m = Model[system.model](states, episodes, system.alpha)

        for i in m:
            predicted = [ i[x] for x in m.states ]
            mse = skm.mean_squared_error(actual, predicted)
            rmse.append(mse)

        outgoing.put((system, rmse))

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=100)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

initargs = (outgoing, incoming, args.states, args.episodes)

with mp.Pool(args.workers, func, initargs):
    td_alpha = np.linspace(0.05, 0.15, 3)
    mc_alpha = np.linspace(0.01, 0.04, 4)

    for (model, alpha) in zip(('td', 'mc'), (td_alpha, mc_alpha)):
        for a in alpha:
            system = System(model, a)
            outgoing.put(system)

    jobs = len(td_alpha) + len(mc_alpha)
    for _ in range(jobs):
        (system, rmse) = incoming.get()
        plt.plot(rmse, label=str(system))
    plt.legend()

    plt.savefig('example-6.2b.png')

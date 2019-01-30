import sys
import csv
import logging
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np

from walk import TemporalDifference

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

Run = cl.namedtuple('Run', 'repetition, alpha, online, steps')

class Steps(list):
    def __init__(self):
        self.extend([ 1, 2, 3, 8, 15, 30, 60, 100, 200, 1000 ])
        self._personalise()
        self.sort()

    def _personalise(self):
        raise NotImplementedError()

class OnlineSteps(Steps):
    def _personalise(self):
        self.append(5)

class OfflineSteps(Steps):
    def _personalise(self):
        self.extend([ 4, 6 ])

def func(incoming, outgoing, episodes, states, gamma):
    states_ = states + 1
    actuals = np.arange(-states_, states_, states_ + 1) / states_
    actuals[0] = actuals[-1] = 0

    while True:
        run = incoming.get()
        logging.info(run)

        td = TemporalDifference(states, episodes, run.alpha, gamma, run.steps)
        for i in td:
            logging.debug(i)
            logging.debug(actuals)

            # mse = np.sum(np.power(np.subtract(i, actuals), 2)) / states
            # rmse = np.sqrt(mse)
            # outgoing.put({ **run._asdict(), 'rmse': rmse })

        outgoing.put(None)

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=19)
arguments.add_argument('--episodes', type=int, default=10)
arguments.add_argument('--repetitions', type=int, default=100)
arguments.add_argument('--alpha-max', type=float, default=1)
arguments.add_argument('--alpha-step', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

initargs = (outgoing, incoming, args.episodes, args.states, args.gamma)
with mp.Pool(args.workers, func, initargs):
    jobs = 0
    for reps in range(args.repetitions):
        for alpha in np.arange(0, args.alpha_max, args.alpha_step):
            for (i, stairs) in enumerate((OnlineSteps, OfflineSteps)):
                for step in stairs():
                    outgoing.put(Run(reps, alpha, bool(i), step))
                    jobs += 1

    writer = None
    while jobs:
        result = incoming.get()
        if result is None:
            jobs -= 1
        else:
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=result.keys())
                writer.writeheader()
            writer.writerow(result)

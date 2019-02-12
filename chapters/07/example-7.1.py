import sys
import csv
import logging
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np

from walk import OnlineUpdate, OfflineUpdate, TemporalDifference

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

Run = cl.namedtuple('Run', 'repetition, alpha, online, steps')

class Experiments:
    def __init__(self):
        self.steps = [ 1, 2, 3, 8, 15, 30, 60, 100, 200, 1000 ]
        self.td = None

    def __iter__(self):
        for i in self.steps:
            yield (i, self.td)

class OnlineExperiments(Experiments):
    def __init__(self):
        super().__init__()
        self.td = OnlineUpdate
        self.steps.append(5)

class OfflineExperiments(Experiments):
    def __init__(self):
        super().__init__()
        self.td = OfflineUpdate
        self.steps.extend([ 4, 6 ])

def func(incoming, outgoing, episodes, states, gamma):
    while True:
        (run, TD) = incoming.get()
        logging.info(run)

        for i in TD(states, episodes, run.alpha, gamma, run.steps):
            # logging.debug(i)
            outgoing.put({ **run._asdict(),
                           'rmse': TemporalDifference.rmse(i)
            })

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
    num = args.alpha_max / args.alpha_step
    assert(num.is_integer())

    jobs = 0
    for rep in range(args.repetitions):
        for alpha in np.linspace(0, args.alpha_max, int(num)):
            for (i, exp) in enumerate((OfflineExperiments, OnlineExperiments)):
                for (steps, td) in exp():
                    run = Run(rep, alpha, i, steps)
                    outgoing.put((run, td))
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

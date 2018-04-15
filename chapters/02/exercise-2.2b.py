import sys
import csv
import logging
import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser

from bandit import Bandit, Result

def run(incoming, outgoing, arms, pulls):
    while True:
        ((epsilon, bandit)) = incoming.get()

        logging.info('{0} {1}'.format(epsilon, bandit))

        b = Bandit(arms, epsilon)
        for (play, action) in enumerate(it.islice(b, 0, pulls)):
            b.do(action)
            optimal = int(b.isoptimal(action))
            outgoing.put(Result(epsilon, bandit, play, action.reward, optimal))
        outgoing.put(None)

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s')

arguments = ArgumentParser()
arguments.add_argument('--bandits', type=int)
arguments.add_argument('--arms', type=int)
arguments.add_argument('--pulls', type=int)
arguments.add_argument('--epsilon', action='append')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

initargs = (outgoing, incoming, args.arms, args.pulls)

with mp.Pool(args.workers, run, initargs) as pool:
    jobs = 0
    for i in map(float, args.epsilon):
        for j in range(args.bandits):
            outgoing.put((i, j))
            jobs += 1

    writer = csv.DictWriter(sys.stdout, fieldnames=Result._fields)
    writer.writeheader()
    while jobs:
        play = incoming.get()
        if play is None:
            jobs -= 1
        else:
            writer.writerow(play._asdict())

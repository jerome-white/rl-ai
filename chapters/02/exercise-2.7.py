import sys
import csv
import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

from action import Arm
from sstrat import Explore
from bandit import ActionRewardBandit as Bandit

Result = cl.namedtuple('Result', 'bandit, play, alpha, reward, optimal')

def run(incoming, outgoing, arms, pulls):
    while True:
        (bid, step) = incoming.get()
        logging.info('{0}: {1}'.format(bid, bargs))

        arms = [ Arm(x, reward=0, alpha=step) for x in range(arms) ]
        bandit = Bandit(arms, Explore(), epsilon=0.1)

        for (play, arm) in enumerate(it.islice(bandit, 0, pulls)):
            reward = bandit.pull(arm)
            optimal = int(bandit.isoptimal(arm))
            outgoing.put(Result(bid, play, step, reward, optimal))

            for i in arms:
                i.reward += random.choice([-1, 1])
        outgoing.put(None)

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--bandits', type=int)
arguments.add_argument('--arms', type=int)
arguments.add_argument('--pulls', type=int)
arguments.add_argument('--step-size', type=float, action='append')
arguments.add_argument('--epsilon', type=float, action='append')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

initargs = (outgoing, incoming, args.arms, args.pulls)

with mp.Pool(args.workers, run, initargs) as pool:
    jobs = 0
    for i in range(args.bandits):
        for j in args.step_size:
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

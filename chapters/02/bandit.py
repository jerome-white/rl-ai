import sys
import csv
import numpy as np
import logging
import operator as op
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

Result = cl.namedtuple('Result', 'epsilon, bandit, play, reward, optimal')

class Action:
    def __init__(self, name):
        self.name = name
        self.reward = np.random.randn()
        self.estimate = 0
        self.chosen = 0

    def __eq__(self, other):
        return self.name == other.name

    def activate(self):
        self.chosen += 1
        self.estimate += 1 / self.chosen * (self.reward - self.estimate)

class Bandit:
    def __init__(self, arms, epsilon=0):
        self.arms = arms
        self.epsilon = epsilon

        self.plays = 0
        self.points = 0
        self.actions = [ Action(x) for x in range(self.arms) ]

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.actions) # explore
        else:
            action = max(self.actions, key=op.attrgetter('estimate')) # exploit

        return action

    def do(self, action):
        plays = self.plays + 1
        self.points = (self.plays * self.points + action.reward) / plays
        self.plays = plays

        action.activate()

    def isoptimal(self, action):
        return action == max(self.actions, key=op.attrgetter('reward'))

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

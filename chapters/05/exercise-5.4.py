import csv
import random
import logging
import operator as op
import itertools as it
import functools as ft
import collections as cl
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

State = cl.namedtuple('State', 'position, velocity')
_Vector = cl.namedtuple('_Vector', 'x, y')

class Vector(_Vector):
    def __new__(cls, x, y):
        super(Vector, cls).__new__(cls, x, y)

    def __add__(self, other):
        return type(self)(it.starmap(op.add, zip(self, other)))

    def __sub__(self, other):
        return type(self)(it.starmap(op.sub, zip(self, other)))

    def __gt__(self, other):
        return all(it.starmap(op.gt, zip(self, other)))

    def __bool__(self):
        return self.x or self.y

    def clip(self):
        return type(self)(it.starmap(lambda x: np.clip(x, 0, 4), self))

class Track:
    def __init__(self, track, start='s', finish='f', out='.'):
        self.track = []
        self.start = set()
        self.finish = set()

        for track.open() as fp:
            reader = csv.reader()
            for line in reader:
                i = reader.line_num - 1
                row = []
                for (j, cell) in enumerate(line):
                    inbounds = cell != out
                    row.append(inbounds)
                    if inbounds:
                        c = Vector(i, j)
                        if cell == start:
                            self.start.add(c)
                        elif cell == finish:
                            self.finish.add(c)
                self.track.append(row)

    def __getitem__(self, key):
        assert(type(key) == Vector)
        return self.track[key.x][key.y]

    def navigate(position, velocity):
        iterable = it.product(*map(lambda x: range(x + 1), velocity))
        for i in it.starmap(Vector, filter(lambda x: any(x), iterable)):
            pos = position + i
            inbounds = all([ x >= 0 for x in pos ]) and self.track[pos]
            yield (pos, inbounds)

class Race:
    def __init__(self, state, track, reward=-1, penalty=4):
        self.state = state
        self.track = track
        self.reward = reward
        self.penalty = penalty

    def __iter__(self):
        return self

    def __next__(self):
        #
        # Select and take an action
        #
        while True:
            action = Vector(*[ random.randint(-1, 1) for _ in range(2) ])
            velocity = (self.state.velocity + action).clip()
            if velocity:
                break

        route = list(self.track.navigate(self.state.position, velocity))
        (position, inbounds) = max(route, key=op.itemgetter(0))

        if not inbounds:
            reward = -5
        elif not all(map(op.itemgetter(1), route)):
            elegible = [ x for (x, y) in route if y ]
            if not elegible:
                for (i, j) in route:
                    if j and not any(i - self.state.position):
                        elegible.append(j)
                assert(elegible)
            position = max(elegible)
            reward = -5
        else:
            reward = -1

        self.state = State(position, velocity)

        return (self.state, action, reward)

logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--track', type=Path)
args = arguments.parse_args()

# returns = cl.defaultdict(list)
# values = cl.defaultdict(float)
# policy = {}

# for i in range(args.games):
#     st = next(state)

#     #
#     # generate epsiode
#     #
#     player = ft.partial(GreedyPlayer, policy=policy)
#     blackjack = Blackjack(st, player)
#     (episode, reward) = blackjack.play()

#     #
#     # calculate returns
#     #
#     for (j, e) in enumerate(episode):
#         returns[e].append(reward)
#         values[e] = np.mean(returns[e])

#     #
#     # calculate optimal policies
#     #
#     for (s, a) in episode:
#         vals = [ values[(s, x)] for x in range(2) ]
#         best = np.argwhere(vals == np.max(vals))
#         policy[s] = np.random.choice(best.flatten())

#         logging.debug('{}: {} a:{} r:{} - pi:{} Q:{}'
#                       .format(i, s, a, reward, policy[s], vals))

#     logging.info('{0}: {1} -> {2:2d}'.format(i, blackjack, reward))

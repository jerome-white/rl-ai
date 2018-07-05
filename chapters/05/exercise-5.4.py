import csv
import random
import logging
import itertools as it
import functools as ft
import collections as cl
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

_Vector = cl.namedtuple('_Vector', 'x, y')

class Vector(_Vector):
    def __new__(cls, x, y):
        super(Vector, cls).__new__(cls, x, y)

    def __add__(self, other):
        components = []
        for (i, j) in zip(self, other):
            dimension = max(0, min(i + j, 5))
            components.append(dimension)

        return type(self)(*components)

    def __bool__(self):
        return self.x or self.y

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

        if any([ x < 0 for x in key ]) or not self.track[key.x][key.y]:
            raise IndexError(key)

        return self.track[key.x][key.y]

class Car:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

class Race:
    def __init__(self, car, track):
        self.car = car
        self.track = track

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            action = Vector(*[ random.randint(-1, 1) for _ in range(2) ])
            self.velocity += action
            if self.velocity:
                break


        try:
            self.state = self.track[advance]
            reward = -1
            if self.state in self.track.finish:
                raise StopIteration()
        except IndexError:
            reward = -5

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

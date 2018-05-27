import math
import logging
import operator as op
import itertools as it
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'first, second')
Action = cl.namedtuple('Action', 'prob, reward, state')
Inventory = cl.namedtuple('Inventory', 'rented, returned')

class Location:
    def __init__(self, rentals, returns):
        self.params = (rentals, returns)

    def prob(self, inventory):
        return op.mul(*it.starmap(poisson, zip(self.params, inventory)))

class Actions:
    def __init__(self, capacity, locations, profit, cost, movable):
        self.capacity = capacity
        self.profit = profit
        self.cost = cost
        self.movable = movable
        (self.first, self.second) = locations

    def __iter__(self):
        yield from range(-self.movable, self.movable + 1)

    def positions(self, cars, moved):
        product = it.product(range(self.capacity + 1),
                             range(self.capacity + 1))
        for i in it.starmap(Inventory, product):
            total = i.returned - (i.rented + moved)
            if cars + total == self.capacity:
                yield i

    def at(self, state, moved):
        for i in self.positions(state.first, moved):
            first = state.first + moved
            prob = self.first.prob(i)
            reward = self.profit * i.returned + self.cost * abs(moved)

            for j in self.positions(state.second, -moved):
                second = state.second - moved

                p = prob * self.second.prob(j)
                r = reward + self.profit * j.returned
                s = State(first, second)

                yield Action(p, r, s)

class States:
    def __init__(self, capacity, locations):
        self.product = it.product(range(capacity + 1), repeat=len(locations))

    def __iter__(self):
        yield from it.starmap(State, self.product)

def bellman(actions, estimate, discount):
    value = 0
    for a in actions:
        value += a.prob * (a.reward + discount * estimate[a.state])

    return value

def poisson(lam, n):
    return lam ** n / math.factorial(n) * math.e ** -lam

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
arguments.add_argument('--discount', type=float)
arguments.add_argument('--improvement-threshold', type=float)
args = arguments.parse_args()

config = ConfigParser()
config.read(args.config)

#
# Convert configuration options to integers
#
env = cl.defaultdict(dict)
for (i, j) in config.items():
    for (k, v) in j.items():
        env[i][k] = int(v)

capacity = env['system']['capacity']
movable = env['system']['movable']

#
# Establish setup objects
#
locations = []
for (i, j) in env.items():
    if i.startswith('location:'):
        loc = Location(j['requests'], j['returns'])
        locations.append(loc)

#
# Object initialization
#
states = States(capacity, locations)
actions = Actions(capacity,
                  locations,
                  env['cost']['rental'],
                  env['cost']['move'],
                  movable)

values = np.zeros((capacity + 1, ) * len(locations))
policy = np.zeros_like(values, int)
evolution = []

#
# Run!
#
stable = False
while not stable:
    #
    # policy evaluation
    #
    logging.info('policy evaluation')

    delta = np.inf
    while delta > args.improvement_threshold:
        values_ = np.zeros_like(values)
        for s in states:
            a = actions.at(s, policy[s])
            values_[s] = bellman(a, values, args.discount)
        delta = np.sum(np.abs(values_ - values))
        logging.info('delta {0}'.format(delta))
        values = values_

    evolution.append([ sns.heatmap(values, vmin=-movable, vmax=movable) ])

    #
    # policy improvement
    #
    logging.info('policy improvement')

    stable = True
    for s in states:
        optimal = np.argmax([
            bellman(actions.at(s, x), values, args.discount) for x in actions
        ])
        if policy[s] != optimal:
            logging.info('not stable')
            policy[s] = optimal
            stable = False

ani = ArtistAnimation(plt.gcf(), evolution, interval=50, blit=True)
ani.save('exercise-4.4.mp4')

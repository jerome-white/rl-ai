import math
import logging
import operator as op
import itertools as it
import collections as cl
import multiprocessing as mp
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
Transition = cl.namedtuple('Transition', 'state, action')
Inventory = cl.namedtuple('Inventory', 'rented, returned')
Facility = cl.namedtuple('Facility',
                         'capacity, profit, cost, movable, locations')

def poisson(lam, n):
    return lam ** n / math.factorial(n) * math.e ** -lam

def irange(stop):
    yield from range(stop + 1)

def bellman(incoming, outgoing, facility, discount):
    actions = Actions(facility)

    while True:
        (t, v) = incoming.get()

        reward = 0
        for i in actions.at(*t):
            reward += i.prob * (i.reward + discount * v[i.state])
        logging.debug('{0} {1} {2}'.format(s, a, reward))

        outgoing.put((t, reward))

class Location:
    def __init__(self, rentals, returns):
        self.params = (rentals, returns)

    def prob(self, inventory):
        return op.mul(*it.starmap(poisson, zip(self.params, inventory)))

class Actions:
    def __init__(self, facility):
        self.facility = facility
        (self.first, self.second) = facility.locations

    def __iter__(self):
        yield from range(-self.facility.movable, self.facility.movable + 1)

    def positions(self, cars, moved):
        cars -= moved
        if cars >= 0:
            rentable = irange(cars)
            returnable = irange(self.facility.capacity - cars)

            yield from it.starmap(Inventory, it.product(rentable, returnable))

    def at(self, state, action):
        for i in self.positions(state.first, action):
            first = state.first + i.returned - i.rented - action
            prob = self.first.prob(i)
            reward = self.facility.profit * i.returned
            reward += self.facility.cost * abs(action)

            for j in self.positions(state.second, -action):
                second = state.second + j.returned - j.rented + action

                p = prob * self.second.prob(j)
                r = reward + self.facility.profit * j.returned
                s = State(first, second)

                logging.debug('{s1} a:{a} f:{f} s:{s} {s2}'.format(s1=state,
                                                                   a=action,
                                                                   f=first,
                                                                   s=second,
                                                                   s2=s))
                yield Action(p, r, s)

class States:
    def __init__(self, facility):
        self.capacity = facility.capacity
        self.repeat = len(facility.locations)

    def __iter__(self):
        product = it.product(irange(self.capacity), repeat=self.repeat)
        yield from it.starmap(State, product)

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
arguments.add_argument('--discount', type=float)
arguments.add_argument('--improvement-threshold', type=float)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
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

facility = Facility(env['system']['capacity'],
                    env['cost']['rental'],
                    env['cost']['move'],
                    env['system']['movable'],
                    locations)
incoming = mp.Queue()
outgoing = mp.Queue()
initargs = (outgoing, incoming, facility, args.discount)

with mp.Pool(args.workers, bellman, initargs):
    #
    # Object initialization
    #
    states = States(facility)

    values = np.zeros((facility.capacity + 1, ) * len(locations))
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

        while True:
            values_ = np.zeros_like(values)

            for s in states:
                transition = Transition(s, policy[s])
                outgoing.put((transition, values))
            for _ in states:
                (t, r) = incoming.get()
                values_[t.state] = r

            delta = np.sum(np.abs(values_ - values))
            logging.info('delta {0}'.format(delta))

            if delta < args.improvement_threshold:
                break
            values = values_

        ax = sns.heatmap(values, vmin=-facility.movable, vmax=facility.movable)
        evolution.append([ ax ])

        #
        # policy improvement
        #
        logging.info('policy improvement')

        stable = True
        for s in states:
            b = policy[s]

            jobs = 0
            for a in actions:
                if a != b:
                    transition = Transition(s, a)
                    outgoing.put((transition, values))
                    jobs += 1
            optimal = values[s]
            for _ in range(jobs):
                (t, r) = incoming.get()
                if r > optimal:
                    optimal = r
                    policy[s] = t.action

            if b != policy[s]:
                log.info('unstable')
                stable = False

ani = ArtistAnimation(plt.gcf(), evolution, interval=50, blit=True)
ani.save('exercise-4.4.mp4')

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

def irange(stop):
    yield from range(stop + 1)

def bellman(incoming, outgoing, facility, discount):
    actions = Explorer(facility)

    while True:
        (t, v) = incoming.get()

        reward = 0
        for i in actions.at(*t):
            reward += i.prob * (i.reward + discount * v[i.state])
        logging.debug('{0} {1} {2}'.format(*t, reward))

        outgoing.put((t, reward))

class Location:
    def __init__(self, rentals, returns):
        self.params = (rentals, returns)
        self.computed = {}

    def probability(self, inventory):
        return op.mul(*it.starmap(self.poisson, zip(inventory, self.params)))

    def poisson(self, n, lam):
        key = (n, lam)
        if key not in self.computed:
            self.computed[key] = lam ** n / math.factorial(n) * math.exp(-lam)

        return self.computed[key]

class Explorer:
    def __init__(self, env):
        self.env = env
        (self.first, self.second) = env.locations

    def positions(self, cars, moved):
        cars -= moved
        if cars >= 0:
            rentable = irange(cars)
            returnable = irange(self.env.capacity - cars)

            yield from it.starmap(Inventory, it.product(rentable, returnable))

    def at(self, state, action):
        for i in self.positions(state.first, action):
            first = state.first + i.returned - i.rented - action
            prob = self.first.probability(i)
            reward = self.env.profit * i.returned
            reward += self.env.cost * abs(action)

            for j in self.positions(state.second, -action):
                second = state.second + j.returned - j.rented + action

                p = prob * self.second.probability(j)
                r = reward + self.env.profit * j.returned
                s = State(first, second)

                logging.debug('{s1} a:{a} f:{f} s:{s} {s2}'.format(s1=state,
                                                                   a=action,
                                                                   f=first,
                                                                   s=second,
                                                                   s2=s))
                yield Action(p, r, s)

class Environment:
    def __init__(self, config):
        self.capacity = int(config['system']['capacity'])
        self.movable = int(config['system']['movable'])
        self.profit = float(config['cost']['rental'])
        self.cost = float(config['cost']['move'])

        self.locations = []
        for (i, j) in config.items():
            if i.startswith('location:'):
                l = Location(*[ int(j[x]) for x in ('requests', 'returns') ])
                self.locations.append(l)

    def states(self):
        product = it.product(irange(self.capacity), repeat=len(self.locations))

        yield from it.starmap(State, product)

    def actions(self):
        yield from range(-self.movable, self.movable + 1)

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
arguments.add_argument('--discount', type=float)
arguments.add_argument('--improvement-threshold', type=float)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

config = ConfigParser()
config.read(args.config)

env = Environment(config)

incoming = mp.Queue()
outgoing = mp.Queue()
initargs = (outgoing, incoming, env, args.discount)

with mp.Pool(args.workers, bellman, initargs):
    values = np.zeros((env.capacity + 1, ) * len(env.locations))
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

            jobs = 0
            for s in env.states():
                transition = Transition(s, policy[s])
                outgoing.put((transition, values))
                jobs += 1
            for _ in range(jobs):
                (t, r) = incoming.get()
                values_[t.state] = r

            delta = np.sum(np.abs(values_ - values))
            logging.info('delta {0}'.format(delta))

            if delta < args.improvement_threshold:
                break
            values = values_

        ax = sns.heatmap(values, vmin=-env.movable, vmax=env.movable)
        evolution.append([ ax ])

        #
        # policy improvement
        #
        logging.info('policy improvement')

        stable = True
        for s in env.states():
            b = policy[s]

            jobs = 0
            for a in env.actions():
                if a != b:
                    transition = Transition(s, a)
                    outgoing.put((transition, values))
                    jobs += 1
            optimal = values[s]
            for _ in range(jobs):
                (t, r) = incoming.get()
                if r > optimal:
                    optimal = r
                    assert(s == t.state)
                    policy[s] = t.action

            if b != policy[s]:
                log.info('unstable')
                stable = False

ani = ArtistAnimation(plt.gcf(), evolution, interval=50, blit=True)
ani.save('exercise-4.4.mp4')

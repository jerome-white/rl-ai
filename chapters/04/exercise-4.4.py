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

logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'first, second')
Action = cl.namedtuple('Action', 'prob, reward, state')
Transition = cl.namedtuple('Transition', 'state, action')
Inventory = cl.namedtuple('Inventory', 'rented, returned')

def poisson_(func):
    computed = {}

    def wrapper(n, lam):
        key = (n, lam)
        if key not in computed:
            computed[key] = func(*key)
        return computed[key]

    return wrapper

@poisson_
def poisson(n, lam):
    return math.pow(lam, n) / math.factorial(n) * math.exp(-lam)

def irange(stop):
    yield from range(stop + 1)

def bellman(incoming, outgoing, env, discount):
    actions = Explorer(env)

    while True:
        (t, v) = incoming.get()
        logging.debug(t)

        reward = 0
        for i in actions.at(*t):
            reward += i.prob * (i.reward + discount * v[i.state])

        outgoing.put((t, reward))

class Location:
    def __init__(self, rentals, returns):
        self.params = (rentals, returns)

    def probability(self, inventory):
        return op.mul(*it.starmap(poisson, zip(self.params, inventory)))

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

class StateEvolution:
    def __init__(self, rewards, policies):
        self.data = []
        self.update(rewards, policies)

    def update(self, rewards, policies):
        for (i, j) in zip(self.data, (rewards, policies)):
            i.append(np.copy(j))

    def write(self):
        names = map(lambda x: Path('rental-' + x), ('rewards', 'policies'))
        for (i, j) in zip(names, self.data):
            np.savez_compressed(i, *j)

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
    reward = np.zeros((env.capacity + 1, ) * len(env.locations))
    policy = np.zeros_like(reward, int)
    evolution = StateEvolution(reward, policy)

    #
    # Run!
    #
    stable = False
    for i in it.takewhile(lambda _: not stable, it.count()):
        logging.critical('iteration {0}'.format(i))

        #
        # policy evaluation
        #
        logging.warning('policy evaluation')

        while True:
            reward_ = np.zeros_like(reward)

            jobs = 0
            for s in env.states():
                transition = Transition(s, policy[s])
                outgoing.put((transition, reward))
                jobs += 1
            for _ in range(jobs):
                (t, r) = incoming.get()
                reward_[t.state] = r

            delta = np.sum(np.abs(reward_ - reward))
            logging.info('delta {0}'.format(delta))

            if delta < args.improvement_threshold:
                break
            reward = reward_

        #
        # policy improvement
        #
        logging.warning('policy improvement')

        stable = True
        for s in env.states():
            b = policy[s]

            jobs = 0
            for a in env.actions():
                if a != b:
                    transition = Transition(s, a)
                    outgoing.put((transition, reward))
                    jobs += 1
            optimal = reward[s]
            for _ in range(jobs):
                (t, r) = incoming.get()
                if r > optimal:
                    optimal = r
                    assert(s == t.state)
                    policy[s] = t.action

            if b != policy[s]:
                stable = False
        logging.warning('stable: {0}'.format(stable))

        evolution.update(reward, policy)
    evolution.write()

import math
import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'first, second')
Action = cl.namedtuple('Action', 'prob, reward, state')
Transition = cl.namedtuple('Transition', 'state, action')
Observation = cl.namedtuple('Observation', 'probability, profit')

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
    cache = {}
    actions = Explorer(env)

    while True:
        (t, v) = incoming.get()

        if t not in cache:
            cache[t] = list(actions.explore(t.state, t.action))

        reward = 0
        for i in cache[t]:
            reward += i.prob * (i.reward + discount * v[i.state])

        outgoing.put((t, reward))

class Location:
    def __init__(self, requests, returns, capacity):
        self.requests = requests
        self.returns = returns
        self.capacity = capacity

        self.rewards = {}
        self.transitions = {}

    def service(self, cars):
        for requested in irange(self.capacity):
            yield (requested, min(cars, requested))

    def reward(self, cars, profit=1):
        if cars not in self.rewards:
            rwd = 0
            for (requested, rented) in self.service(cars):
                rwd += rented * poisson(requested, self.requests)
            rwd *= profit
            self.rewards[cars] = rwd

        return self.rewards[cars]

    def transition(self, start, end):
        situation = (start, end)

        if situation not in self.transitions:
            probability = 0

            for (requested, rented) in self.service(start):
                for returned in irange(self.capacity):
                    available = start + returned - rented
                    available = max(0, min(self.capacity, available))
                    if available == end:
                        rq = poisson(requested, self.requests)
                        rt = poisson(returned, self.returns)
                        probability += rq * rt

            self.transitions[situation] = probability

        return self.transitions[situation]

class Explorer:
    def __init__(self, env):
        self.env = env
        (self.first, self.second) = env.locations

    def explore_(self, state, action):
        r = self.env.cost * abs(action)
        r += self.first.reward(state.first, self.env.profit)
        r += self.second.reward(state.second, self.env.profit)

        for s in self.env.states():
            p = self.first.transition(state.first, s.first)
            p *= self.second.transition(state.second, s.second)

            yield Action(p, r, s)

    def explore(self, state, action):
        s = State(state.first + action, state.second - action)
        if all([ 0 <= x <= self.env.capacity for x in s ]):
            yield from self.explore_(s, action)

class Environment:
    def __init__(self, config):
        self.capacity = int(config['system']['capacity'])
        self.movable = int(config['system']['movable'])
        self.profit = float(config['cost']['rental'])
        self.cost = float(config['cost']['move'])

        self.locations = []
        for (i, j) in config.items():
            if i.startswith('location:'):
                expected = [ int(j[x]) for x in ('requests', 'returns') ]
                location = Location(*expected, self.capacity)
                self.locations.append(location)

    def states(self):
        product = it.product(irange(self.capacity), repeat=len(self.locations))

        yield from it.starmap(State, product)

    def actions(self):
        yield from range(-self.movable, self.movable + 1)

class StateEvolution:
    def __init__(self, rewards, policies):
        self.data = ([], [])
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
    step = 0
    stable = False
    while not stable:
        logging.info('iteration {0}'.format(step))

        #
        # policy evaluation
        #
        logging.info('policy evaluation')

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
            logging.debug('delta {0}'.format(delta))
            if delta < args.improvement_threshold:
                break

            reward = reward_

        #
        # policy improvement
        #
        logging.info('policy improvement')

        stable = True
        for s in env.states():
            (b, optimal) = [ x[s] for x in (policy, reward) ]

            jobs = 0
            for a in env.actions():
                if a != b:
                    transition = Transition(s, a)
                    outgoing.put((transition, reward))
                    jobs += 1
            for _ in range(jobs):
                (t, r) = incoming.get()
                if r > optimal:
                    optimal = r
                    assert(s == t.state)
                    policy[s] = t.action

            if b != policy[s]:
                stable = False

        logging.info('stable: {0}'.format(stable))

        evolution.update(reward, policy)
        step += 1
    evolution.write()

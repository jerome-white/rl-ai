import math
import logging
import itertools as it
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'first, second')
Action = cl.namedtuple('Action', 'prob, reward, state')
Dynamic = cl.namedtuple('Dynamic', 'rented, returned, moved')

class Location:
    def __init__(self, rentals, returns):
        self.rentals = rentals
        self.returns = returns

    def prob(self, dynamic):
        rentals = Location.poisson(self.rentals, abs(dynamic.rented))
        returned = Location.poisson(self.returns, dynamic.returned)

        return rentals * returned

    @staticmethod
    def poisson(lam, n):
        return lam ** n / math.factorial(n) * math.e ** -lam

class Policy:
    def __init__(self, capacity, movable, profit, cost):
        self.capacity = capacity
        self.movable = movable
        self.profit = profit
        self.cost = cost

    def evaluate(self, state, first, second):
        for i in actions(state.first, self.capacity, irange(self.movable)):
            first_ = state.first + i.moved
            partial_reward = self.profit * abs(i.rented) + self.cost * i.moved

            for j in actions(state.second, self.capacity, -i.moved):
                logging.debug('{0} {1}'.format(state.first, i))
                logging.debug('{0} {1}'.format(state.second, j))

                prob = first.prob(i) * second.prob(j)
                reward = partial_reward + self.profit * abs(j.rented)
                state_ = State(first_, state.second - i.moved)

                yield Action(prob, reward, state_)

class Bellman:
    def __init__(self, policy, discount, first, second):
        self.policy = policy
        self.discount = discount
        self.first = first
        self.second = second

    def evaluate(state, estimate):
        for e in self.policy.evaluate(state, self.first, self.second):
            existing = estimate[e.state.first][e.state.second]
            yield e.prob * (e.reward + self.discount * existing)

def irange(stop):
    yield from range(stop + 1)

def states(capacity, locations):
    product = it.product(irange(capacity), repeat=locations)

    yield from it.starmap(State, product)

def actions(cars, capacity, movable):
    try:
        iter(movable)
    except TypeError:
        movable = (movable, )

    rentable = range(-capacity, 1)
    returnable = irange(capacity)

    for i in it.starmap(Dynamic, it.product(rentable, returnable, movable)):
        if capacity - sum(i) == cars:
            yield i

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
arguments.add_argument('--discount', type=float)
arguments.add_argument('--improvement-threshold', type=float)
args = arguments.parse_args()

config = ConfigParser()
config.read(args.config)

# Convert configuration options to integers
env = cl.defaultdict(dict)
for (i, j) in config.items():
    for (k, v) in j.items():
        env[i][k] = int(v)

capacity = env['system']['capacity']

# Establish setup objects
policy = Policy(capacity,
                env['system']['movable'],
                env['cost']['rental'],
                env['cost']['move'])

locations = []
for (i, j) in env.items():
    if i.startswith('location:'):
        loc = Location(j['requests'], j['returns'])
        locations.append(loc)

# run
for s in states(capacity, len(locations)):
    for e in policy.evaluate(s, *locations):
        print(e)

bellman = Bellman(policy, args.discount, *locations)
backup = np.zeros()

while True:
    #
    # policy evaluation
    #
    delta = None

    while delta is None or delta > args.improvement_threshold:
        backup_ = np.zeros()
        for s in states:
            rewards = bellman.evaluate(s, backup)
            backup_[s.first][s.second] = sum(rewards)
        delta = np.sum(np.abs(backup_ - backup))
        backup = backup_

    # policy improvement
    stable = True
    for s in states:
        b = pi(s)

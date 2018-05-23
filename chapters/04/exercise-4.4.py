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

def irange(stop):
    yield from range(stop + 1)

def states(capacity):
    yield from it.starmap(State, it.product(irange(capacity), repeat=2))

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

def evaluate(state, first, second):
    for i in actions(state.first, 20, irange(5)):
        first_ = state.first + i.moved
        partial_reward = 10 * abs(i.rented) - 2 * i.moved

        for j in actions(state.second, 20, -i.moved):
            logging.debug('{0} {1}'.format(state.first, i))
            logging.debug('{0} {1}'.format(state.second, j))

            prob = first.prob(i) * second.prob(j)
            reward = partial_reward + 10 * abs(j.rented)
            state_ = State(first_, state.second - i.moved)

            yield Action(prob, reward, state_)

first = Location(3, 3)
second = Location(4, 2)
for s in states(20):
    for e in evaluate(s, first, second):
        print(e)

# arguments = ArgumentParser()
# arguments.add_argument('--config', type=Path)
# arguments.add_argument('--discount', type=float)
# args = arguments.parse_args()

# config = ConfigParser()
# config.read(args.config)

# env = cl.defaultdict(dict)
# for (i, j) in config.items():
#     for (k, v) in j.items():
#         env[i][k] = float(v)

# syscfg = env['system']
# system = System(syscfg['capacity'], syscfg['movable'], 2)

# while True:
#     # policy evaluation
#     while True:
#         delta = 0
#         for s in states:
#             pass

#     # policy improvement
#     stable = True
#     for s in states:
#         b = pi(s)

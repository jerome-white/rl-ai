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
Inventory = cl.namedtuple('Inventory', 'rented, returned, moved')

class Location:
    def __init__(self, rentals, returns):
        self.rentals = rentals
        self.returns = returns

    def prob(self, dynamic):
        rentals = poisson(self.rentals, abs(dynamic.rented))
        returned = poisson(self.returns, dynamic.returned)

        return rentals * returned

class Actions:
    def __init__(self, capacity, locations, profit, cost, movable):
        self.capacity = capacity
        (self.first, self.second) = locations
        self.profit = profit
        self.cost = cost
        self.movable

    def __iter__(self):
        yield from range(-self.movable, self.movable + 1)

    def positions(self, cars, moves):
        rents = range(-self.capacity, 1)
        returns = range(self.capacity + 1)

        for i in it.product(rents, returns):
            inventory = Inventory(*i, moves)
            if self.capacity - sum(inventory) == cars:
                yield inventory

    def at(self, state, moves):
        for i in self.positions(state.first, moves):
            first_ = state.first + i.moved
            partial_reward = self.profit * abs(i.rented) + self.cost * i.moved

            for j in self.positions(state.second, -i.moved):
                logging.debug('{0} {1}'.format(state.first, i))
                logging.debug('{0} {1}'.format(state.second, j))

                prob = self.first.prob(i) * self.second.prob(j)
                reward = partial_reward + self.profit * abs(j.rented)
                state_ = State(first_, state.second - i.moved)

                yield Action(prob, reward, state_)

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
policy = Policy(capacity,
                env['system']['movable'],
                env['cost']['rental'],
                env['cost']['move'])

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

values = np.zeros((capacity, ) * len(locations))
policy = np.zeros_like(values)

#
# Run!
#
stable = False
while not stable:
    #
    # policy evaluation
    #
    delta = np.inf
    while delta > args.improvement_threshold:
        values_ = np.zeros_like(values)
        for s in states:
            a = actions.at(s, policy[s])
            values_[s] = bellman(a, values, args.discount)
        delta = np.sum(np.abs(values_ - values))
        values = values_

    #
    # policy improvement
    #
    stable = True
    for s in states:
        optimal = np.argmax([
            bellman(actions.at(s, x), values, args.discount) for x in actions
        ])
        if policy[s] != optimal:
            policy[s] = optimal
            stable = False

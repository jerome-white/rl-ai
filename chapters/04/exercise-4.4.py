import math
import itertools as it
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

Location = cl.namedtuple('Location', 'cars, requests, returns')
State = cl.namedtuple('State', 'first, second')
Action = cl.namedtuple('Action', 'prob, reward, state')

class System:
    def __init__(self, cars, moves, locations):
        self.cars = cars
        self.moves = moves
        self.locations = locations

    def __iter__(self):
        yield from it.product(irange(self.cars),
                              irange(self.cars),
                              irange(self.moves))
#                               repeat=self.locations)

def irange(stop):
    yield from range(int(stop) + 1)

def poisson(lam, n):
    return lam ** n / math.factorial(n) * math.e ** -lam

def states(system, locations):
    for cars in it.combinations_with_replacement(irange(system.cars),
                                                 len(locations)):
        locs = []
        for (i, j) in zip(cars, locations):
            locs.append(Location(i, j['requests'], j['returns']))

        yield State(*locs)

def actions(state, system, costs):
    for (rented, returned, moved) in system:
        net = state.first.cars - rented + returned - moved
        if 0 <= net <= system.cars:
            prob = poisson(state.first.requests, rented)
            prob *= poisson(state.first.returns, returned)
            reward = rented * costs['rental'] + moved * costs['move']

            yield Action(prob, reward, None)

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
arguments.add_argument('--discount', type=float)
args = arguments.parse_args()

config = ConfigParser()
config.read(args.config)

env = cl.defaultdict(dict)
for (i, j) in config.items():
    for (k, v) in j.items():
        env[i][k] = float(v)

syscfg = env['system']
system = System(syscfg['capacity'], syscfg['movable'], 2)

for s in states(system, (env['first'], env['second'])):
    for a in actions(s, system, env['cost']):
        print(a)

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

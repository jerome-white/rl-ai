import itertools as it
from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

class States:
    def __init__(self, cars, locations):
        self.cars = cars
        self.locations = locations

    def __iter__(self):
        cars = range(self.cars)
        yield from it.combinations_with_replacement(cars, self.locations)

class Actions:
    def __init__(self, locations):
        self.locations = locations

    def __iter__(self):
        for rented in range(20):
            reward = rented * self.credit
            for returned in range(20 - rented):
                for moved in range(5):
                    reward - moved * self.cost

arguments = ArgumentParser()
arguments.add_argument('--config', type=Path)
args = arguments.parse_args()

config = ConfigParser()
config.read(str(args.config))

states = States(int(config['capacity']), len(config) - 1)

while True:
    # policy evaluation
    while True:
        delta = 0
        for s in states:
            pass

    # policy improvement
    stable = True
    for s in states:
        b = pi(s)

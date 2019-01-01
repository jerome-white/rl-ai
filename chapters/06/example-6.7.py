import math
import random
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'servers, customer')

class Server:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self):
        return sum([ np.random.binomial(1, self.p) for _ in range(self.n) ])

class Customer:
    def __init__(self, n, h, priority=None):
        self.n = n
        self.h = h
        self.priority = priority

        low = self.n - 1
        self.weights = [ self.h / low ] * low + [ self.h ]

    def __float__(self):
        return math.pow(2, int(self.priority))

    def __int__(self):
        if self.priority is None:
            raise AttributeError()
        return self.priority

    def __str__(self):
        return '{} ({})'.format(int(self), float(self))

    def __next__(self):
        priority = random.choices(range(self.n), weights=self.weights).pop()
        return type(self)(self.n, self.h, priority)

arguments = ArgumentParser()
arguments.add_argument('--servers', type=int, default=10)
arguments.add_argument('--high-priority', type=float, default=0.5)
arguments.add_argument('--release-rate', type=float, default=0.06)
# arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

Q = np.zeros((args.servers, 4, 2))

servers = Server(args.servers, args.release_rate)
customer = Customer(4, args.high_priority)

for i in it.count():
    s = State(servers(), next(customer))
    logging.info('{} {} {}'.format(i, s.servers, s.customer))

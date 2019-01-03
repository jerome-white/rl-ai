import math
import random
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

_State = cl.namedtuple('State', 'servers, customer')

class State(_State):
    def __new__(cls, row, column):
        return super(State, cls).__new__(cls, servers, customer)

    def __bool__(self):
        return bool(self.servers)

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
        priority = random.choices(range(self.n), weights=self.weights)
        return type(self)(self.n, self.h, priority[0])

def transition(servers, customer):
    state = None

    for _ in range(2):
        s = servers() if state is None else max(state.servers - 1, 0)
        c = next(customer)

        yield State(s, int(c))

def choose(state, explore=True):
    if explore and np.random.binomial(1, args.epsilon):
        action = random.randrange(len(state))
    else:
        action = np.argwhere(state == np.max(state)).flatten()

    return np.random.choice(action)

arguments = ArgumentParser()
arguments.add_argument('-n', type=int, default=10,
                       help='Number of servers')
arguments.add_argument('-h', type=float, default=0.5,
                       help='Proportion of high priority customers')
arguments.add_argument('-p', type=float, default=0.06,
                       help='Probability of a server becoming free')
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--beta', type=float, default=0.01)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--steps', type=int, default=2e6)
# arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

servers = Server(args.n, args.p)
customer = Customer(4, args.h)

Q = np.zeros((args.n, 4, 2))
rho = 0

for i in range(args.steps):
    # Establish the state
    (state, state_) = transition(servers, customer)

    logging.info('{}: {} -> {}'.format(i, state, state_))

    # Choose the action
    action = choose(Q[state])

    # Take the action and observe the reward. (The subsequent state
    # doesn't depend on the chosen action.)
    reward = 2 ** state.customer if state and action else 0

    action_ = choose(Q[state_], False)
    target = Q[(*state_, action_)] - Q[(*state, action)]
    Q[(*state, action)] += self.alpha * (reward - rho + target)

    if Q[(*state, action)] == choose(Q[state], False):
        rho += args.beta * (reward - rho + target)

logging.critical('rho {}'.format(rho))

import math
import random
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

#
#
#
def flatten(lst):
    for i in [ lst ]:
        try:
            for j in i:
                yield from flatten(j)
        except TypeError:
            yield i

#
#
#
_State = cl.namedtuple('State', 'servers, customer')

class State(_State):
    def __new__(cls, servers, customer):
        return super(State, cls).__new__(cls, servers, customer)

    def __bool__(self):
        return bool(self.servers)

    def __float__(self):
        return math.pow(2, self.customer)

    def __str__(self):
        return 's:{0} c:{1}'.format(*self)

#
#
#
class Servers:
    def __init__(self, n, p):
        self.p = p

        self.free = True
        self.busy = not self.free
        self.status = [ self.free ] * n

    def __call__(self):
        return sum(self.status)

    def __len__(self):
        return len(self.status)

    def engage(self, action):
        i = 0
        for _ in range(action):
            try:
                i = self.status.index(self.free, i)
                self.status[i] = self.busy
                i += 1
            except ValueError:
                break

    def allocate(self):
        freed = 0

        for i in range(len(self)):
            if self.status[i] == self.busy and random.random() <= self.p:
                self.status[i] = self.free
                freed += 1

        return freed

class Customers:
    def __init__(self, n, h):
        self.n = n

        low = self.n - 1
        self.weights = [ (1 - h) / low ] * low + [ h ]

    def __len__(self):
        return self.n

    def __next__(self):
        return random.choices(range(len(self)), weights=self.weights).pop()

#
#
#
class Policy:
    def __init__(self, servers, customers, actions=2):
        self.q = np.zeros((servers, customers, actions))

    def __getitem__(self, item):
        return self.q[tuple(flatten(item))]

    def __setitem__(self, key, value):
        self.q[key] = value

    def choose(self, state, epsilon):
        if not state:
            decision = 0
        elif np.random.binomial(1, epsilon):
            actions = len(self[state])
            decision = random.randrange(actions)
        else:
            decision = self.greedy(state)

        return decision

    def greedy(self, state):
        ptr = self[state]
        action = np.argwhere(ptr == np.max(ptr)).flatten()

        return np.random.choice(action)

    def isgreedy(self, state, action):
        return self[(state, action)] == self.greedy(state)

class System:
    def __init__(self, servers, customer):
        self.servers = servers
        self.customer = customer

    def step(self, action=None):
        self.servers.allocate()
        if action is not None:
            self.servers.engage(action)
        free = max(servers() - 1, 0)

        return State(free, next(self.customer))

#
#
#
arguments = ArgumentParser()
arguments.add_argument('--high-priority', type=float, default=0.5)
arguments.add_argument('--p-free', type=float, default=0.06)
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--beta', type=float, default=0.01)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--steps', type=int, default=int(2e6))
args = arguments.parse_args()

servers = Servers(10, args.p_free)
customers = Customers(4, args.high_priority)
system = System(servers, customers)

Q = Policy(len(servers), len(customers))
rho = 0

state = system.step()
for i in range(args.steps):
    action = Q.choose(state, args.epsilon)
    reward = float(state) if action else 0
    state_ = system.step(action)

    logging.info('{}: {} -[a:{} r:{}]-> {}'
                 .format(i, state, action, reward, state_))

    action_ = Q.greedy(state_)
    target = reward - rho + Q[(state_, action_)] - Q[(state, action)]
    Q[(state, action)] += args.alpha * target

    if Q.isgreedy(state, action):
        rho += args.beta * target
        logging.debug('updated rho: {}'.format(rho))

    state = state_

logging.critical('rho {}'.format(rho))

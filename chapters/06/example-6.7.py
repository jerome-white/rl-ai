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

def transition(servers, customer):
    state = None

    for _ in range(2):
        s = servers() if state is None else max(state.servers - 1, 0)
        c = next(customer)

        yield State(s, c)
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
        return ' '.join(map(str, self))

#
#
#
class Server:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self):
        return sum([ np.random.binomial(1, self.p) for _ in range(self.n) ])

class Customer:
    def __init__(self, n, h):
        self.n = n

        low = self.n - 1
        self.weights = [ h / low ] * low + [ h ]

    def __next__(self):
        return random.choices(range(self.n), weights=self.weights).pop()

#
#
#
class Policy:
    def __init__(self, servers, customers=4, actions=2):
        self.q = np.zeros((servers, customers, actions))

    def __getitem__(self, item):
        return self.q[tuple(flatten(item))]

    def __setitem__(self, key, value):
        self.q[key] = value

    def choose(self, state, epsilon):
        if np.random.binomial(1, epsilon):
            actions = len(self[state])
            return random.randrange(actions)
        else:
            return self.greedy(state)

    def greedy(self, state):
        ptr = self.q[state]
        action = np.argwhere(ptr == np.max(ptr)).flatten()

        return np.random.choice(action)

    def isgreedy(self, state, action):
        return self[(state, action)] == self.greedy(state)

#
#
#
arguments = ArgumentParser()
arguments.add_argument('--servers', type=int, default=10)
arguments.add_argument('--high-priority', type=float, default=0.5)
arguments.add_argument('--p-free', type=float, default=0.06)
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--beta', type=float, default=0.01)
arguments.add_argument('--epsilon', type=float, default=0.1)
arguments.add_argument('--steps', type=int, default=int(2e6))
# arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

_customers = 4
servers = Server(args.servers, args.p_free)
customer = Customer(_customers, args.high_priority)

Q = Policy(args.servers)
rho = 0

for i in range(args.steps):
    # Establish the state
    (state, state_) = transition(servers, customer)
    logging.info('{}: {} -> {}'.format(i, state, state_))

    # Choose the action
    action = Q.choose(state, args.epsilon)

    # Take the action and observe the reward. (The subsequent state
    # doesn't depend on the chosen action.)
    reward = float(state) if state and action else 0

    action_ = Q.greedy(state_)
    target = Q[(state_, action_)] - Q[(state, action)]
    Q[(state, action)] += args.alpha * (reward - rho + target)

    if Q.isgreedy(state, action):
        rho += args.beta * (reward - rho + target)

logging.critical('rho {}'.format(rho))

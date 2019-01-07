import random
import itertools as it
import collections as cl

import numpy as np
import pandas as pd

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

    def __int__(self):
        return 2 ** self.customer

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
        low = n - 1
        self.weights = [ (1 - h) / low ] * low + [ h ]

    def __len__(self):
        return len(self.weights)

    def __next__(self):
        return random.choices(range(len(self)), weights=self.weights).pop()

#
#
#
class Policy:
    def __init__(self, servers, customers, actions=2, accounting=False):
        shape = (servers + 1, customers, actions)
        self.q = np.zeros(shape)

        self.accounting = {}
        if accounting:
            for (s, c, a) in it.product(*map(range, shape)):
                if not s and a:
                    continue
                self.accounting[(s, c, a)] = 0

    def __getitem__(self, item):
        return self.q[tuple(flatten(item))]

    def __setitem__(self, key, value):
        key = tuple(flatten(key))
        if self.accounting:
            self.accounting[key] += 1
        self.q[key] = value

    def __bool__(self):
        if not self.accounting:
            raise AttributeError()
        return all(self.accounting.values())

    def greedy(self, state):
        ptr = self[state]
        action = np.argwhere(ptr == np.max(ptr)).flatten()

        return np.random.choice(action)

    def choose(self, state, epsilon):
        if not state:
            action = 0
        elif np.random.binomial(1, epsilon):
            action = random.randrange(len(self[state]))
        else:
            action = self.greedy(state)

        return action

    def toarray(self, f):
        for (i, row) in enumerate(self.q):
            for (j, cell) in enumerate(row):
                s = State(i, j)
                yield (s.servers, int(s), f(cell))

    def toframe(self, f):
        columns=('servers', 'priority', 'value')
        return pd.DataFrame(self.toarray(f), columns=columns)

class System:
    def __init__(self, servers, customer):
        self.servers = servers
        self.customer = customer
        self.state = None

    def step(self, action=None):
        self.servers.allocate()

        reward = 0
        if action:
            self.servers.engage(action)
            if self.state:
                reward = int(self.state)
        self.state = State(self.servers(), next(self.customer))

        return (reward, self.state)

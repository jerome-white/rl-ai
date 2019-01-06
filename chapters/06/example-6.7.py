import random
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    def __init__(self, servers, customers, actions=2):
        self.q = np.zeros((servers + 1, customers, actions))

    def __getitem__(self, item):
        return self.q[tuple(flatten(item))]

    def __setitem__(self, key, value):
        self.q[key] = value

    def greedy(self, state):
        ptr = self[state]
        action = np.argwhere(ptr == np.max(ptr)).flatten()

        return np.random.choice(action)

    def isgreedy(self, state, action):
        return self[(state, action)] == self.greedy(state)

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

    def step(self, action=None):
        self.servers.allocate()
        if action:
            self.servers.engage(action)

        return State(self.servers(), next(self.customer))

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
    reward = int(state) if action else 0
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

print('rho {}'.format(rho))

#
# Priority versus number of free servers (Figure 6.17, top)
#
title = '$\\rho \\approx${}'.format(round(rho, 2))

df = Q.toframe(np.argmax)
df = df[df['servers'] > 0].pivot(index='priority',
                                 columns='servers',
                                 values='value')
sns.heatmap(df, vmin=0, vmax=1, cmap='BrBG')
plt.title('Policy ({})'.format(title))
plt.savefig('figure-6.17a.png')

plt.clf()

#
# Value of the best action versus number of free servers (Figure 6.17,
# bottom)
#
df = Q.toframe(np.max)

# sns.lineplot(x='servers',
#              y='value',
#              hue='priority',
#              data=df)

for (i, g) in df.groupby('priority'):
    plt.plot(g['servers'], g['value'], label=i)
plt.grid(True)
plt.title('Value Function ({})'.format(title))
plt.legend(title='Priority')
plt.xticks(range(len(servers)))
plt.xlabel('Number of free servers')
plt.ylabel('Value of best action')
plt.savefig('figure-6.17b.png')

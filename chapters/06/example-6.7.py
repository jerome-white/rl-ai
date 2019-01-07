import random
import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from queuing import Servers, Customers, System, Policy

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

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

Q = Policy(len(servers), len(customers), accounting=True)
rho = 0

(_, state) = system.step()
for i in range(args.steps):
    action = Q.choose(state, args.epsilon)
    (reward, state_) = system.step(action)

    action_ = Q.greedy(state_)
    update = reward - rho + Q[(state_, action_)]
    Q[(state, action)] += args.alpha * (update - Q[(state, action)])

    logging.info('{0}: {1} -[a:{2} r:{3} (Q:{4})]-> {5}'
                 .format(i,
                         state,
                         action,
                         reward,
                         round(Q[(state, action)], 5),
                         state_))

    action_ = Q.greedy(state)
    if Q[(state, action)] == Q[(state, action_)]:
        rho += args.beta * (update - Q[(state, action_)])
        logging.debug('updated rho: {}'.format(rho))

    state = state_

print('rho', rho)
print('Explored', bool(Q))

# np.save('e67', Q.q)
# for i in Q.accounting.items():
#     print(*i, sep=',')

#
# Priority versus number of free servers (Figure 6.17, top)
#
title = '$\\rho \\approx${}'.format(round(rho, 2))

df = Q.toframe(np.argmax)
df = df[df['servers'] > 0].pivot(index='priority',
                                 columns='servers',
                                 values='value')
sns.heatmap(df, vmin=0, vmax=1, square=True, cmap='RdBu_r', cbar=False)
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
plt.xticks(np.sort(df['servers'].unique()))
plt.xlabel('Number of free servers')
plt.ylabel('Value of best action')
plt.savefig('figure-6.17b.png')

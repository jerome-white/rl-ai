import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import seaborn as sns

from blackjack import Blackjack

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

def func(args):
    (n, fill, ace) = args

    #
    # generate epsiode
    #
    blackjack = Blackjack()
    (episode, reward) = blackjack.play()

    logging.info(('[{0:'+str(fill)+'d}] {1} -> {2:2d}')
                 .format(n, blackjack, reward))

    #
    # calculate returns
    #
    returns = cl.defaultdict(list)
    for (state, _) in it.islice(episode, 1, None):
        if ace ^ state.ace:
            returns[state].append(reward)

    return returns

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
arguments.add_argument('--with-ace', action='store_true')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

with mp.Pool(args.workers) as pool:
    returns = cl.defaultdict(list)

    f = lambda x: (x, len(str(args.games)), args.with_ace)
    for i in pool.imap_unordered(func, map(f, range(args.games))):
        for (k, v) in i.items():
            returns[k].extend(v)

V = np.zeros((21 - 12 + 1, 10 - 1 + 1))
for (k, v) in returns.items():
    V[k.player - 12, k.dealer - 1] = np.mean(v)

ax = sns.heatmap(V, vmin=-1, vmax=1, cmap='BrBG')
ax.invert_yaxis()
ax.set_xticklabels(['A'] + list(range(2, 11)))
ax.set_yticklabels(range(12, 22))
ax.get_figure().savefig('example-5.1.png')

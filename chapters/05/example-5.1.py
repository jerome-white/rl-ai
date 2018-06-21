import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from blackjack import Blackjack

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

class Returns:
    def __init__(self):
        self.aces = [ cl.defaultdict(list) for _ in range(2) ]

    def __iter__(self):
        yield from self.aces

    def __getitem__(self, key):
        assert(type(key) is bool)
        return self.aces[key]

    def __add__(self, other):
        for i in (True, False):
            ptr = self[i]
            for (k, v) in other[i].items():
                ptr[k].extend(v)

        return self

def func(args):
    #
    # generate epsiode
    #
    blackjack = Blackjack()
    (episode, reward) = blackjack.play()

    logging.info('{1} -> {2:2d} [ {0} ]'.format(args, blackjack, reward))

    #
    # calculate returns
    #
    returns = Returns()
    for (state, _) in episode:
        returns[state.ace][state].append(reward)

    return returns

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

with mp.Pool(args.workers) as pool:
    returns = Returns()
    for i in pool.imap_unordered(func, range(args.games)):
        returns += i

for i in (True, False):
    name = 'example-5.1-{0}.png'.format(i)
    V = np.zeros((21 - 12 + 1, 10 - 1 + 1))

    for (k, v) in returns[i].items():
        V[k.player - 12, k.dealer - 1] = np.mean(v)

    plt.clf()
    ax = sns.heatmap(V, vmin=-1, vmax=1, cmap='BrBG')
    ax.invert_yaxis()
    ax.set_xticklabels(['A'] + list(range(2, 11)))
    ax.set_yticklabels(range(12, 22))
    ax.get_figure().savefig(name)

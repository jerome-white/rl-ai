import random
import logging
import itertools as it
import functools as ft
import collections as cl
from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from blackjack import State, Player, Blackjack

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

class StateSpace:
    def __init__(self):
        args = (range(12, 21 + 1), range(1, 10 + 1), (True, False))
        self.args = tuple(map(tuple, args))
        self.shape = tuple(map(len, it.islice(self.args, 2)))

    def __iter__(self):
        yield from it.starmap(State, it.product(*self.args))

    def __next__(self):
        return State(*map(random.choice, self.args))

class GreedyPlayer(Player):
    def __init__(self, policy, value=0, cards=0, ace=False):
        super().__init__(value, cards, ace)
        self.policy = policy

    def hit(self, facecard):
        state = State(self.value, facecard, self.ace)
        if state in self.policy:
            decision = self.policy[state]
        else:
            decision = super().hit(facecard)

        return decision

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
args = arguments.parse_args()

state = StateSpace()

returns = cl.defaultdict(list)
values = cl.defaultdict(float)
policy = {}

for i in range(args.games):
    st = next(state)

    #
    # generate epsiode
    #
    player = ft.partial(GreedyPlayer, policy=policy)
    blackjack = Blackjack(st, player)
    (episode, reward) = blackjack.play()

    logging.info('{1} -> {2:2d} [ {0} ]'.format(i, blackjack, reward))

    #
    # calculate returns
    #
    for e in episode:
        returns[e].append(reward)
        values[e] = np.mean(returns[e])

    #
    # calculate optimal policies
    #
    for (s, _) in episode:
        values = [ values[(s, x)] for x in map(bool, range(2)) ]
        best = np.argwhere(values == np.max(values))
        policy[s] = np.random.choice(best.flatten())

for a in (True, False):
    V = np.zeros(state.shape)
    pi = np.zeros_like(V)

    for s in filter(lambda x: x.ace == a, state):
        index = (s.player - 12, s.dealer - 1)
        V[index] = values[(s, a)]
        pi[index] = policy[s]

    for (i, j) in zip(('V', 'pi'), (V, pi)):
        plt.clf()
        name = 'example-5.3_{0}-{1}.png'.format(i, a)

        ax = sns.heatmap(j, vmin=-1, vmax=1, cmap='BrBG')
        ax.invert_yaxis()
        ax.set_xticklabels(['A'] + list(range(2, 11)))
        ax.set_yticklabels(range(12, 22))
        ax.get_figure().savefig(name)

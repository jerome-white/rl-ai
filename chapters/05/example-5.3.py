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
        self.shape = tuple(map(len, it.islice(self.arg, 2)))

    def __iter__(self):
        yield from it.starmap(State, it.product(*self.args))

    def __next__(self):
        return State(*map(random.choice, self.args))

class GreedyPlayer(Player):
    def __init__(self, Q, value=0, cards=0, ace=False):
        super().__init__(value, cards, ace)
        self.Q = Q

    def stick(self, facecard):
        if self.Q:
            state = State(int(self), facecard, self.ace)
            if state in self.Q:
                args = np.argwhere(self.Q[state] == max(self.Q[state]))
                best = args.flatten().astype(bool)
            else:
                best = (True, False)
            decision = np.random.choice(best)
        else:
            decision = super().stick(facecard)

        return decision

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
args = arguments.parse_args()

returns = cl.defaultdict(lambda: cl.defaultdict(list))
Q = cl.defaultdict(lambda: [ -np.inf ] * len(('hit', 'stick')))
policy = {}

state = StateSpace()

for i in range(args.games):
    s = next(state)

    #
    # generate epsiode
    #
    player = ft.partial(GreedyPlayer, Q=Q)
    blackjack = Blackjack(s, player)
    (episode, reward) = blackjack.play()

    logging.info('{0}: {1} {2}'.format(i, s, reward))

    #
    # calculate returns
    #
    for (s_, a) in episode:
        ptr = returns[s_][a]
        ptr.append(reward)
        Q[s_][a] = np.mean(ptr)

    #
    # calculate optimal policies
    #
    for (s_, _) in episode:
        policy[s_] = np.argmax(Q[s])

for a in (True, False):
    V = np.zeros(state.shape)
    pi = np.zeros_like(V)

    for s in filter(lambda x: x.ace == a, state):
        index = (s.player - 12, s.dealer - 1)
        V[index] = Q[s][a]
        pi[index] = policy[s]

    for (i, j) in zip(('V', 'pi'), (V, pi)):
        plt.clf()
        name = 'example-5.3_{0}-{1}.png'.format(i, a)

        ax = sns.heatmap(j, vmin=-1, vmax=1, cmap='BrBG')
        ax.invert_yaxis()
        ax.set_xticklabels(['A'] + list(range(2, 11)))
        ax.set_yticklabels(range(12, 22))
        ax.get_figure().savefig(name)

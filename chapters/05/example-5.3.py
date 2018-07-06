import random
import logging
import itertools as it
import functools as ft
import collections as cl
from argparse import ArgumentParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import blackjack as bj
from blackjack import State, Player, Blackjack

logging.basicConfig(level=logging.INFO,
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
        self.initial = True

    def hit(self, facecard):
        if self.initial:
            self.initial = False
            action = random.choice((True, False))
        else:
            state = self.tostate(facecard)
            if state in self.policy:
                action = self.policy[state]
            else:
                action = super().hit(facecard)

        return action

arguments = ArgumentParser()
arguments.add_argument('--games', type=int, default=500000)
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

    #
    # calculate returns
    #
    for (j, e) in enumerate(episode):
        returns[e].append(reward)
        values[e] = np.mean(returns[e])

    #
    # calculate optimal policies
    #
    for (s, a) in episode:
        policy[s] = bj.fairmax(values, s)

        logging.debug('{}: {} a:{} r:{} - pi:{} Q:{}'
                      .format(i, s, a, reward, policy[s], vals))

    logging.info('{0}: {1} -> {2:2d}'.format(i, blackjack, reward))

for ace in (True, False):
    V = np.zeros(state.shape)
    pi = np.zeros_like(V)

    for s in filter(lambda x: x.ace == ace, state):
        index = (s.player - 12, s.dealer - 1)
        pi[index] = policy[s]
        V[index] = values[(s, pi[index])]

    for (i, j) in zip(('V', 'pi'), (V, pi)):
        plt.clf()
        name = 'example-5.3_{0}-{1}.png'.format(i, ace)

        ax = sns.heatmap(j, vmin=-1, vmax=1, cmap='BrBG')
        ax.invert_yaxis()
        ax.set_xticklabels(['A'] + list(range(2, 11)))
        ax.set_yticklabels(range(12, 22))
        ax.get_figure().savefig(name)

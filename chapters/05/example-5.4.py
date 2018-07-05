#
# Comes from second edition!
#

import random
import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

from blackjack import State, Player, Blackjack

logging.basicConfig(level=logging.INFO,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

class RandomPlayer(Player):
    def hit(self, facecard):
        return random.choice((True, False))

def func(args):
    (games, gamma) = args

    state = State(13, 2, True)
    actions = (True, False)
    b = len(actions)

    Q = cl.defaultdict(float)
    C = cl.defaultdict(int)

    rewards = []
    weights = []

    for i in range(games):
        blackjack = Blackjack(state, RandomPlayer)
        (episode, reward) = blackjack.play()

        G = 0
        W = 1

        for e in it.takewhile(lambda _: W, episode):
            G = gamma * G + reward
            C[e] += W
            Q[e] += (W / C[e]) * (G - Q[e])

            (s, a) = e
            player = Player(s.player, 2, s.ace)
            p = player.hit(s.dealer) == a
            W *= p / b

        rewards.append(W * G)
        weights.append(W)

    r = sum(rewards)

    return ( r / x for x in (len(rewards), sum(weights)) )

arguments = ArgumentParser()
arguments.add_argument('--games', type=int, default=500000)
arguments.add_argument('--runs', type=int, default=100)
arguments.add_argument('--gamma', type=int, default=1)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

with mp.Pool(args.workers) as pool:
    iterable = it.repeat((args.games, args.gamma), range(args.runs))
    for (i, j) in pool.imap_unordered(func, iterable):
        pass

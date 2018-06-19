import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np

from blackjack import State, Blackjack

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

def states():
    while True:
        args = (range(12, 21 + 1), range(1, 10 + 1), (True, False))
        yield State(*map(random.choice, args))

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
arguments.add_argument('--with-ace', action='store_true')
args = arguments.parse_args()

Q = cl.defaultdict(lambda: [ -np.inf ] * len(('hit', 'stick')))
policy = {}

for (i, state) in zip(range(args.games), states()):
    #
    # generate epsiode
    #
    blackjack = Blackjack(state)
    (episode, reward) = blackjack.play()

    logging.info('{0}: {1} {2}'.format(i, state, reward))

    #
    # calculate returns
    #
    for (state, action) in episode:
        Q[state][action].append(reward)

    #
    # calculate optimal policies
    #
    for (state, _) in episode:
        policy[state] = np.argmax(Q[state])

V = np.zeros((21 - 12 + 1, 10 - 2 + 1))
for (k, v) in values.items():
    if args.with_ace ^ v.ace:
        V[k.player, k.dealer] = np.mean(v)

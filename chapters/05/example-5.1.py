import logging
import collections as cl
from argparse import ArgumentParser

import numpy as np
import matplotlib as plt

from blackjack import Blackjack

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
arguments.add_argument('--with-ace', action='store_true')
args = arguments.parse_args()

returns = cl.defaultdict(list)

for i in range(args.games):
    #
    # generate epsiode
    #
    blackjack = Blackjack()
    (episode, reward) = blackjack.play()

    logging.info('{0}: {1}'.format(i, reward))

    #
    # calculate returns
    #
    for (state, _) in episode:
        returns[state].append(reward)

V = np.zeros((21 - 12 + 1, 10 - 1 + 1))
for (k, v) in returns.items():
    if args.with_ace ^ k.ace:
        V[k.player - 12, k.dealer - 1] = np.mean(v)

plt.imshow(V)
plt.savefig('example-5.1.png')

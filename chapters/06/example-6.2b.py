import sys
import csv
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np

import walk

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=1)
arguments.add_argument('--alpha', type=float, default=1)
arguments.add_argument('--gamma', type=float, default=1)
args = arguments.parse_args()

V = {}
returns = cl.defaultdict(list)
states = [ chr(ord('A') + x) for x in range(args.states) ]

writer = csv.DictWriter(sys.stdout,
                        fieldnames=states,
                        extrasaction='ignore')
writer.writeheader()

for i in range(args.episodes):
    reward = None
    episode = list(walk.walk(states))

    for (j, ep) in enumerate(reversed(episode)):
        if j > 0:
            logging.debug(ep)
        
            if reward is None:
                reward = ep.reward

            r = returns[ep.state]
            r.append(reward)
            V[ep.state] = np.mean(r)

    writer.writerow(V)

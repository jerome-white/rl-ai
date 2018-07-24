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

states = []
V = cl.defaultdict(float)
for i in range(args.states):
    s = chr(ord('A') + i)
    states.append(s)
    V[s] = 0.5

writer = csv.DictWriter(sys.stdout,
                        fieldnames=states,
                        extrasaction='ignore')
writer.writeheader()

for i in range(args.episodes):
    episode = list(walk.walk(states))
    episode.pop()

    reward = episode[-1].reward
    for ep in reversed(episode):
        logging.debug(ep)
        V[ep.state] += args.alpha * (args.gamma * reward - V[ep.state])

    writer.writerow(V)

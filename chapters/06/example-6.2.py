import sys
import csv
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

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
V = cl.defaultdict(int)
for i in range(args.states):
    s = chr(65 + i)
    states.append(s)
    V[s] = 0.5

writer = csv.DictWriter(sys.stdout,
                        fieldnames=states,
                        extrasaction='ignore')
writer.writeheader()
writer.writerow(V)

for i in range(args.episodes):
    s = None
    for s_ in walk.walk(states):
        if s is not None:
            value = s.reward + args.gamma * V[s_.state] - V[s.state]
            logging.debug('{} {} {}'.format(s, s_, value))
            V[s.state] += args.alpha * value
        s = s_
    logging.info(dict(V))
    writer.writerow(V)

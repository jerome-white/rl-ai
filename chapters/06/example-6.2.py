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
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--alpha', type=float, default=1)
args = arguments.parse_args()

states = args.states + 1
V = cl.defaultdict(lambda: 0.5)

fieldnames = [ chr(65 + x) for x in range(args.states) ]
writer = csv.DictWriter(sys.stdout,
                        fieldnames=fieldnames,
                        restval=0.5,
                        extrasaction='ignore')
writer.writeheader()
writer.writerow(V)

for i in range(args.episodes):
    s = None
    for s_ in walk.walk(states):
        if s is not None:
            value = s.reward + args.gamma * V[s_.state] - V[s.state]
            V[s.state] += args.alpha * value
        s = s_

    writer.writerow(V)

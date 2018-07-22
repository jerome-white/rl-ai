import sys
import csv
import math
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

import numpy as np
import sklearn.metrics as skm

import walk

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=1)
arguments.add_argument('--alpha', type=float, default=1)
arguments.add_argument('--gamma', type=float, default=1)
arguments.add_argument('--initial', type=float, default=0.5)
args = arguments.parse_args()

states = [ chr(65 + x) for x in range(args.states) ]

V = cl.defaultdict(lambda: args.initial)
actual = [ x / (len(states) + 1) for x in range(1, len(states) + 1) ]

fieldnames = [
    *states,
    'rmse',
]
writer = csv.DictWriter(sys.stdout,
                        fieldnames=fieldnames,
                        restval=args.initial,
                        extrasaction='ignore')
writer.writeheader()

predicted = [ V[x] for x in states ]
rmse = math.sqrt(skm.mean_squared_error(actual, predicted))
writer.writerow({ **V, 'rmse': rmse })

for i in range(args.episodes):
    s = None
    for s_ in walk.walk(states):
        if s is not None:
            value = s.reward + args.gamma * V[s_.state] - V[s.state]
            logging.debug('{} {} {}'.format(s, s_, value))
            V[s.state] += args.alpha * value
        s = s_
    logging.info(dict(V))

    predicted = [ V[x] for x in states ]
    rmse = math.sqrt(skm.mean_squared_error(actual, predicted))
    writer.writerow({ **V, 'rmse': rmse })

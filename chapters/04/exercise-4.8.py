# import logging
import operator as op
# import itertools as it
# import collections as cl
# import multiprocessing as mp
# from pathlib import Path
from argparse import ArgumentParser

import numpy as np

Action = cl.namedtuple('Action', 'prob, reward, state')

def explore(state, action, value):
    for i in (op.pos, op.neg):
        a = i(action)
        p = 0.4 if a > 0 else 1 - 0.4
        state + a

arguments = ArgumentParser()
arguments.add_argument('--improvement-threshold', type=float)
# arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

V = [ 0 ] * 100 + [ 1 ]
delta = None

while delta is None or delta > args.improvement_threshold:
    for s in range(101):
        rewards = []
        for a in range(min(s, 100 - s) + 1):
            traj = s + a
            rewards.append(args.heads * V[traj] + (1 - args.heads) * V[traj])
        v = max(rewards)
        delta = max(delta, abs(v - V[s])
        V[s] = v

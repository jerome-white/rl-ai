import logging
import operator as op
# import multiprocessing as mp
# from pathlib import Path
from argparse import ArgumentParser

class Optimal:
    def __init__(self, action=None, reward=None):
        self.action = action
        self.reward = reward

    def __bool__(self):
        return self.action is not None and self.reward is not None

    def __gt__(self, other):
        return self.reward > other.reward

arguments = ArgumentParser()
arguments.add_argument('--heads', type=float, default=0.4)
arguments.add_argument('--improvement-threshold', type=float)
arguments.add_argument('--maximum-capital', type=int, default=100)
# arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

V = [ 0 ] * args.maximum_capital + [ 1 ]

delta = None
while delta is None or delta > args.improvement_threshold:
    for s in range(len(V)):
        rewards = []
        for a in range(min(s, args.maximum_capital - s) + 1):
            value = V[s + a]
            rewards.append(args.heads * value + (1 - args.heads) * value)
        v = max(rewards)
        delta = max(delta, abs(v - V[s]))
        V[s] = v

policy = [ None ]
for s in range(1, len(V) - 1):
    optimal = Optimal()
    for a in range(min(s, args.maximum_capital - s) + 1):
        value = V[s + a]
        reward = args.heads * value + (1 - args.heads) * value
        current = Optimal(a, reward)
        if not optimal or current > optimal:
            optimal = current
    policy.append(optimal.action)

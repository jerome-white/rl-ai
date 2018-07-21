import itertools as it
from argparse import ArgumentParser

import walk

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=4)
args = arguments.parse_args()

states = args.states + 2
V = np.zeros(states + 1)

s = None
for s_ in walk.walk(states):
    if s is not None:
        value = (s.reward + args.gamma * V[s_.state] - V[s.state])
        V[s.state] += args.alpha * value
    s = s_

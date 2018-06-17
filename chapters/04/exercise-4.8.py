# import logging
import itertools as it
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np

from util import StateEvolution

class Transition:
    def __init__(self, action, state, reward):
        self.action = action
        self.state = state
        self.reward = reward

    def __gt__(self, other):
        return self.reward > other.reward

class States:
    def __init__(self, heads, capital):
        self.heads = heads
        self.capital = capital

    def at(self, values):
        for s in range(len(values)):
            yield (s, values, self.heads, self.capital)

def bellman(args):
    (state, values, heads, capital) = args

    optimal = None
    for action in range(min(state, capital - state) + 1):
        reward = heads * values[state + action]
        reward += (1 - heads) * values[state - action]

        current = Transition(action, state, reward)
        if optimal is None or current > optimal:
            optimal = current

    return optimal

arguments = ArgumentParser()
arguments.add_argument('--heads', type=float, default=0.4)
arguments.add_argument('--improvement-threshold', type=float, default=1e-9)
arguments.add_argument('--maximum-capital', type=int, default=100)
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

with mp.Pool(args.workers) as pool:
    states = States(args.heads, args.maximum_capital)

    V = np.zeros(args.maximum_capital + 1)
    V[-1] = 1
    sweeps = StateEvolution(V)

    while True:
        delta = 0
        for i in pool.imap_unordered(bellman, states.at(np.copy(V))):
            delta = max(delta, abs(i.reward - V[i.state]))
            V[i.state] = i.reward
        sweeps.update(V)        
        if delta < args.improvement_threshold:
            break


    sweeps.write('coin-values')

    policy = np.zeros_like(V)                 
    sweeps = StateEvolution(policy)
            
    iterable = it.islice(states.at(V), 1, len(V) - 1)
    for i in pool.imap_unordered(bellman, iterable):
        policy[i.state] = i.action
        sweeps.update(policy)

    sweeps.write('coin-policy')

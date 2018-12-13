import random
import operator as op
import itertools as it
import collections as cl

import numpy as np

State_ = cl.namedtuple('State_', 'row, column')

class State(State_):
    def __new__(cls, row, column):
        return super(State, cls).__new__(cls, row, column)

    def __str__(self):
        return '{0},{1}'.format(self.row, self.column)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return type(self)(*it.starmap(op.add, zip(self, other)))

    def inbounds(self, bounds):
        return all([ 0 <= x < y for (x, y) in zip(self, bounds) ])

class Policy:
    def __init__(self, bounds):
        self.bounds = bounds

    def neighbors(self, state):
        for i in it.permutations(range(-1, 2), r=2):
            action = State(*i)
            if op.xor(*map(abs, action)):
                s = state + action
                if s.inbounds(self.bounds):
                    yield action

    def choose(self, state, Q):
        raise NotImplementedError()

class EpsilonGreedyPolicy(Policy):
    def __init__(self, bounds, epsilon):
        super().__init__(bounds)
        self.epsilon = epsilon

    def choose(self, state, Q):
        if np.random.binomial(1, self.epsilon):
            actions = list(self.neighbors(state))
        else:
            best = None
            actions = []

            for i in self.neighbors(state):
                s = state + i
                if best is None or Q[s] >= best:
                    actions.append(i)

        return random.choice(actions)

class Grid:
    def __init__(self, shape, goal):
        self.shape = State(*shape)
        self.goal = goal

    def walk(self, state, action):
        state_ = self.blow(state + action)
        if not state_.inbounds(self.shape):
            state_ = state

        reward = -int(state != self.goal)

        return (state_, reward)

    def blow(self, state):
        return state

class WindyGrid(Grid):
    def __init__(self, shape, goal):
        super().__init__(shape, goal)
        self.speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def blow(self, state):
        shift = State(0, self.speeds[state.column])
        return state + shift

import random
import operator as op
import itertools as it
import collections as cl

import numpy as np

#
#
#

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

#
#
#

class Q:
    def __init__(self, grid):
        self.q = {}

        for i in it.product(grid.walk(), grid.actions()):
            self[i] = 0

    def __getitem__(self, item):
        (state, action) = item
        return self.q[state][action]

    def __setitem__(self, key, value):
        (state, action) = key
        if state not in self.q:
            self.q[state] = {}
        self.q[state][action] = value

    def select(self, state):
        raise NotImplementedError()

class EpsilonGreedyPolicy(Q):
    def __init__(self, grid, epsilon):
        super().__init__(grid)
        self.epsilon = epsilon

    def select(self, state):
        s = self.q[state]

        if np.random.binomial(1, self.epsilon):
            actions = list(s.keys())
        else:
            actions = []
            jackpot = -np.inf

            for (action, reward) in s.items():
                if reward >= jackpot:
                    if reward > jackpot:
                        actions.clear()
                        jackpot = reward
                    actions.append(action)

        return random.choice(actions)

#
#
#

class GridWorld:
    def __init__(self, rows, columns, goal, compass, wind=None):
        self.shape = State(rows, columns)
        self.goal = goal
        self.compass = compass
        self.wind = wind

    def walk(self):
        yield from it.starmap(State, it.product(*map(range, self.shape)))

    def navigate(self, state, action):
        state_ = state + action
        if state_.inbounds(self.shape):
            state = state_

        if self.wind is not None:
            state_ = state + self.wind.blow(state)
            if state_.inbounds(self.shape):
                state = state_

        reward = -int(state != self.goal)

        return (state, reward)

    def actions(self, state=None):
        for action in self.compass:
            if state is not None:
                state_ = state + action
                if not state_.inbounds(self.shape):
                    continue
            yield action

#
#
#

class Compass:
    def __init__(self):
        self.around = range(-1, 2)

    def __iter__(self):
        raise NotImplementedError()

class FourPointCompass(Compass):
    def __iter__(self):
        for i in it.permutations(self.around, r=2):
            if op.xor(*map(abs, i)):
                yield i

class KingsMoves(Compass):
    def __init__(self, stationary=False):
        super().__init__()
        self.stationary = stationary

    def __iter__(self):
        for i in it.product(self.around, repeat=2):
            if self.stationary or any(i):
                yield i

#
#
#

class Wind:
    def __init__(self):
        columns = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.speeds = list(map(op.neg, columns))

    def blow(self, state):
        return State(self.speeds[state.column], 0)

class StochasticWind(Wind):
    def blow(self, state):
        state_ = super().blow(state)
        increment = random.choice(range(-1, 2))

        return state_ + State(increment, 0)

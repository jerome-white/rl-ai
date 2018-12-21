import random
import operator as op
import itertools as it
import collections as cl

import numpy as np

#
# A State is a position on the grid
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

    def amax(self, state):
        best = []
        target = -np.inf
        actions = self.q[state]

        for (action, reward) in actions.items():
            if reward >= target:
                if reward > target:
                    best.clear()
                    target = reward
                best.append(action)

        return random.choice(best)

    def select(self, state):
        raise NotImplementedError()

class EpsilonGreedyPolicy(Q):
    def __init__(self, grid, epsilon):
        super().__init__(grid)
        self.epsilon = epsilon

    def select(self, state):
        if np.random.binomial(1, self.epsilon):
            actions = self.q[state].keys()
            return random.choice(list(actions))
        else:
            return self.amax(state)

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

    def inbounds(self, state):
        return all([ 0 <= x < y for (x, y) in zip(state, self.shape) ])

    def navigate(self, state, action):
        for f in (lambda _: action, self.wind.blow):
            action_ = State(*f(state))
            state_ = state + action_
            if self.inbounds(state_):
                state = state_

        reward = -int(state != self.goal)

        return (state, reward)

    def actions(self, state=None):
        for action in self.compass:
            if state is not None:
                state_ = state + State(*action)
                if not self.inbounds(state_):
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

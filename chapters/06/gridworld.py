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
    def __init__(self, grid):
        self.grid = grid

    def choose(self, state, Q):
        raise NotImplementedError()

class EpsilonGreedyPolicy(Policy):
    def __init__(self, grid, epsilon):
        super().__init__(grid)
        self.epsilon = epsilon

    def choose(self, state, Q):
        actions = self.grid.actions(state)

        if np.random.binomial(1, self.epsilon):
            elegible = list(actions)
        else:
            best = None
            elegible = []

            for action in actions:
                state_ = state + action
                if best is None or Q[state_] >= best:
                    elegible.append(action)

        return random.choice(elegible)

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

    def actions(self, state):
        navigation = it.permutations(range(-1, 2), r=2)

        for action in it.starmap(State, navigation):
            if self.legal(action):
                state_ = state + action
                if state_.inbounds(self.shape):
                    yield action

    def blow(self, state):
        raise NotImplementedError()

    def legal(self, action):
        raise NotImplementedError()

class WindyGrid(Grid):
    def __init__(self, shape, goal):
        super().__init__(shape, goal)
        self.speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def blow(self, state):
        # shift = State(0, self.speeds[state.column])
        # return state + shift
        return state

    def legal(self, action):
        return op.xor(*map(abs, action))

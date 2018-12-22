import random
import operator as op
import itertools as it
import collections as cl

import numpy as np

def sarsa(grid, start, Q, alpha, gamma):
    for episode in it.count():
        step = 0
        state = start
        action = Q.select(state)

        while state != grid.goal:
            (state_, reward) = grid.navigate(state, action)
            action_ = Q.select(state_)

            now = (state, action)
            later = (state_, action_)

            Q[now] += alpha * (reward + gamma * Q[later] - Q[now])

            (state, action) = later

            yield (episode, step, reward)
            step += 1

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

class Policy:
    def __init__(self, grid):
        self.q = {}

        for state in grid:
            for action in grid.actions(state):
                self[(state, action)] = 0

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

class EpsilonGreedyPolicy(Policy):
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
    def __init__(self, shape, goal, compass, wind):
        self.shape = State(*shape)
        self.goal = goal
        self.compass = compass
        self.wind = wind

    def __iter__(self):
        yield from it.starmap(State, it.product(*map(range, self.shape)))

    def actions(self, state=None):
        for action in self.compass:
            if state is not None:
                state_ = state + State(*action)
                if not self.inbounds(state_):
                    continue
            yield action

    def inbounds(self, state):
        return all([ 0 <= x < y for (x, y) in zip(state, self.shape) ])

    def navigate(self, state, action):
        for f in (lambda _: action, self.wind.blow):
            state_ = state + State(*f(state))
            if self.inbounds(state_):
                state = state_

        reward = -int(state != self.goal)

        return (state, reward)

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

class KingsMovesNinth(Compass):
    def __iter__(self):
        yield from it.product(self.around, repeat=2)

class KingsMoves(KingsMovesNinth):
    def __iter__(self):
        yield from filter(lambda x: any(x), super().__iter__())

#
#
#

class Wind:
    def __init__(self):
        columns = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.speeds = list(map(op.neg, columns))

    def blow(self, state):
        return (self.speeds[state.column], 0)

class StochasticWind(Wind):
    def blow(self, state):
        (movement, ) = super().blow(state)
        movement += random.choice(range(-1, 2))

        return (movement, 0)

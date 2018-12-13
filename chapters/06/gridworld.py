import random
import itertools as it
import collections as cl

Action = cl.namedtuple('Action', 'state, reward')
State_ = cl.namedtuple('State_', 'row, column')

class State(State_):
    def __new__(cls, row, column):
        return super(State, cls).__new__(cls, row, column)

    def __str__(self):
        return '{0},{1}'.format(self.row, self.column)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return type(self)(it.starmap(op.add, zip(self, other)))

    def inbounds(self, bounds):
        return all([ 0 <= x < y for (x, y) in zip(s, bounds) ])

    def neighbors(self, bounds):
        for i in it.permutations(range(-1, 2), r=2):
            if not any(i):
                s = self + type(self)(*i)
                if self.inbounds(bounds):
                    yield s

class Policy:
    def __init__(self, bounds):
        self.bounds = bounds

    def potential(self, state):
        yield from state.neighbors(self.bounds):

    def choose(self, state, Q):
        raise NotImplementedError()

class RandomPolicy(Policy):
    def choose(self, state, Q):
        return random.choice(self.potential(state))

class Grid:
    def __init__(self, start, goal, shape):
        self.state = start
        self.goal = goal
        self.shape = shape

    def __bool__(self):
        return self.state != self.goal

    def __int__(self):
        return -int(self)

    def walk(self, action):
        previous = self.state

        self.state = self.blow(self.state + action)
        if not self.state.inbounds(self.shape):
            self.state = previous

        return (self.state, int(self))

    def blow(self, state):
        raise NotImplementedError()

class WindyGrid(Grid):
    def __init__(self, start, goal, shape):
        super().__init__(start, goal, shape)
        self.speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def blow(self, state):
        shift = State(0, self.speeds[state.column])
        return state + shift

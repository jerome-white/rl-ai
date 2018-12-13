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

    def inbounds(self, xbound, ybound):
        return 0 <= self.row < xbound and 0 <= self.column < ybound

    def neighbors(self):
        for i in it.permutations(range(-1, 2), 2):
            if not any(i):
                yield self + type(self)(*i)

class Grid:
    def __init__(self, rows, columns, policy):
        self.grid = cl.defaultdict(set)

        self.start = start
        self.goal = goal
        self.policy = policy

        for i in it.product(range(rows), range(columns), repeat=2):
            state = State(*i)
            for t in state.neighbors():
                s = t if t.inbounds(rows, columns) else state
                r = 0 if t == goal else -1
                self.grid[state].add(Action(s, r))

        self.dimensions = (rows, columns)
        self.state = None

    def __iter__(self):
        self.state = self.start
        return self

    def __next__(self):
        if self.state == self.goal:
            raise StopIteration()

        action = self.policy.select(self.state)
        self.state = action.state

        return action

    def walk(self, state, action):
        return self.state[state][action]

class WindyGrid(Grid):
    def __init__(self):
        super().__init__()
        self.speeds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def __getitem__(self, item):
        return item + State(0, self._move(item.column))

    def _move(self, col):
        return self.speeds[col]

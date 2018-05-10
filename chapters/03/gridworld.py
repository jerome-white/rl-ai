import random
import itertools as it
import collections as cl

def navigator():
    for (i, j) in it.permutations(range(-1, 2), 2):
        if not i or not j:
            yield lambda x, y: (x + i, y + j)

Action = cl.namedtuple('Action', 'state, reward')
State_ = cl.namedtuple('State_', 'x, y')

class State(State_):
    def __new__(cls, x, y):
        return super(State, cls).__new__(cls, x, y)

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def inbounds(self, xbound, ybound):
        return 0 <= self.x < xbound and 0 <= self.y < ybound

    def __str__(self):
        return '{0},{1}'.format(self.x, self.y)

class Estimate(State):
    def __init__(self, x, y):
        super().__init__(self, x, y)
        self.estimate = 0

    def __str__(self):
        return '{0:5.2f}'.format(self.estimate)

class Grid:
    def __init__(self, rows, columns=None, S=State):
        if columns is None:
            columns = rows

        self.S = S
        self.grid = cl.defaultdict(list)

        for m in range(rows):
            for n in range(columns):
                s = self.S(m, n)
                for f in navigator():
                    t = self.S(*f(s.x, s.y))
                    if t.inbounds(rows, columns):
                        a = Action(t, 0)
                    else:
                        a = Action(s, -1)
                    self.grid[s].append(a)

    def __iter__(self):
        for s in sorted(self.grid.keys()):
            for t in self.grid[s]:
                yield (s, t)

    def __str__(self):
        x = None
        row = []
        table = []

        for s in sorted(self.grid.keys()):
            cell = str(s)

            if x is None:
                x = s.x
            if x != s.x:
                table.append('  '.join(row))
                row = []
                x = s.x

            row.append(cell)

        if row:
            table.append('  '.join(row))

        return '\n\n'.join(table)

    def walk(self, state=None):
        if state is None:
            state = self.S(0, 0)

        while True:
            position = self.grid[state.x][state.y]
            s = random.choice(position)
            yield s.reward
            state = s.state

class SpecialGrid(Grid):
    def __init__(self):
        super().__init__(5)

        self.grid[State(0, 1)] = [ Action(State(4, 1), 10) ]
        self.grid[State(0, 3)] = [ Action(State(2, 3),  5) ]

class Estimate(list):
    def __init__(self, dimensions, init=0):
        (rows, columns) = dimensions
        for m in range(rows):
            self.append([ init ] * columns)

    def __str__(self):
        sep = None
        table = []

        for row in self:
            line = [ '{0:5.2f}'.format(x) for x in row ]
            if sep is None:
                width = len(line[0]) + 2
                sep = ('+' + '-' * width) * len(self[0]) + '+'
            table.extend([
                sep,
                '| ' + ' | '.join(line) + ' |',
            ])
        if table:
            table.append(sep)

        return '\n'.join(table)

    def __sub__(self, other):
        diff = 0

        for (i, j) in zip(self, other):
            for (x, y) in zip(i, j):
                diff += x - y

        return diff

class BackupSolver:
    def __init__(self, grid, discount):
        self.grid = grid
        self.discount = discount
        self.before = None

    def __iter__(self):
        self.before = Estimate(self.grid.dimensions)
        return self

    def __next__(self):
        after = Estimate(self.grid.dimensions)

        for (state, actions) in self.grid:
            estimates = self.bellman(actions)
            after[state.x][state.y] = self.collect(estimates)

        before = self.before
        self.before = after

        return (before, before - after)

    def bellman(self, actions):
        p = self.probability(actions)
        for a in actions:
            est = self.before[a.state.x][a.state.y]
            yield p * (a.reward + self.discount * est)

    def probability(self, actions):
        raise NotImplementedError()

    def collect(self, actions):
        raise NotImplementedError()

class Estimate(list):
    def __init__(self, dimensions):
        (rows, columns) = dimensions
        for m in range(rows):
            self.append([ 0 ] * columns)

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

    def estimates(self, actions, discount, probability=1):
        for a in actions:
            est = self[a.state.x][a.state.y]
            yield probability * (a.reward + discount * est)

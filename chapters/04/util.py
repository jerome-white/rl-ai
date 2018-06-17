from pathlib import Path

import numpy as np

class StateEvolution:
    def __init__(self, *args):
        self.data = [ [] for _ in args ]
        self.update(args)

    def update(self, *args):
        for (i, j) in zip(self.data, args):
            i.append(np.copy(j))

    def write(self, *args):
        assert(len(args) == len(self.data))

        for (i, j) in zip(map(Path, args), self.data):
            np.savez_compressed(i, *j)

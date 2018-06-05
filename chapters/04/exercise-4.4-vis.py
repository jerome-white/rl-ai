from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, npz, limit=None):
        self.data = []
        self.ax = None

        if limit is None:
            self.vmin = None
            self.vmax = None
        else:
            (self.vmin, self.vmax) = (-limit, limit)

        load = np.load(npz)
        for (_, data) in load.items():
            if limit is None:
                (x, y) = [ f(data) for f in (np.min, np.max) ]
                self.vmin = x if self.vmin is None else min(self.vmin, x)
                self.vmax = y if self.vmax is None else max(self.vmax, y)

            self.data.append(data)

    def __iter__(self):
        yield from enumerate(self.data)

    def func(self, frame):
        (i, data) = frame

        plt.clf()
        ax = sns.heatmap(data, vmin=self.vmin, vmax=self.vmax, cmap='BrBG')
        ax.set_title('Iteration: {}'.format(i))
        ax.invert_yaxis()

        return (ax, )

arguments = ArgumentParser()
arguments.add_argument('--data', type=Path)
arguments.add_argument('--limit', type=int)
arguments.add_argument('--delay', type=int, default=2)
args = arguments.parse_args()

interval = args.delay / constants.milli
animator = Animator(args.data, args.limit)
ani = FuncAnimation(plt.gcf(),
                    animator.func,
                    frames=animator,
                    interval=interval)
ani.save(args.data.with_suffix('.mp4'))

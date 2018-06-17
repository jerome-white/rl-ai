from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, npz, limit):
        self.limit = limit

        load = np.load(npz)
        self.data = [ x for (_, x) in load.items() ]

    def __iter__(self):
        yield from enumerate(self.data)

    def func(self, frame):
        (i, data) = frame

        plt.clf()
        lines = plt.plot(data)
        plt.title('Sweep: {0}'.format(i))
        plt.grid()

        return (lines, )

arguments = ArgumentParser()
arguments.add_argument('--data', type=Path)
arguments.add_argument('--limit', type=int)
arguments.add_argument('--delay', type=int, default=1)
args = arguments.parse_args()

interval = args.delay / constants.milli
animator = Animator(args.data, args.limit)
ani = FuncAnimation(plt.gcf(),
                    animator.func,
                    frames=animator,
                    interval=interval)
ani.save(args.data.with_suffix('.mp4'))

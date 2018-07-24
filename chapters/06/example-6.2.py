import math
import logging
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from walk import TemporalDifference

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

class Animator:
    def __init__(self, model):
        self.model = model

        limit = len(self.model.states) + 1
        self.keyframes = {
            'True values': [ x / limit for x in range(1, limit) ]
        }

        limit = round(math.log(self.model.episodes))
        self.samples = set([ 10 ** x for x in range(limit) ])
        self.samples.add(0)

    def __iter__(self):
        yield from enumerate(self.model)

    def func(self, frame):
        (i, data) = frame

        plt.clf()

        y = [ data[x] for x in self.model.states ]

        if i in self.samples:
            logging.info(i)
            self.keyframes[i] = y
        else:
            plt.plot(y)

        for (label, yval) in self.keyframes.items():
            plt.plot(yval, label=label)

        plt.ylim(0, 1)
        plt.grid(True)
        plt.title('Step ' + str(i))
        plt.xticks(range(len(self.model.states)), self.model.states)
        plt.legend(loc='upper left')

        return (plt.gca(), )

    def init_func(self):
        return (plt.gca(), )

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=100)
arguments.add_argument('--alpha', type=float, default=0.1)
args = arguments.parse_args()

episodes = args.episodes + 1

model = TemporalDifference(args.states, episodes, args.alpha)
animator = Animator(model)
ani = FuncAnimation(plt.gcf(),
                    animator.func,
                    frames=animator,
                    init_func=animator.init_func,
                    interval=50,
                    save_count=episodes)
ani.save('example-6.2.mp4')

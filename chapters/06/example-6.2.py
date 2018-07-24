import logging
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from walk import TemporalDifference

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

def func(frame):
    (i, data) = frame

    logging.info(i)

    states = sorted(data.keys())
    y = [ data[x] for x in states ]
    limit = len(states) + 1
    samples = set([0] + np.logspace(0, 3, 4, dtype=int).tolist())

    plt.clf()

    plt.plot([ data[x] for x in states ], label='Estimate')
    plt.plot([ x / limit for x in range(1, limit) ], label='True')

    plt.grid(True)
    plt.title('Step ' + str(i))
    plt.xticks(range(len(states)), states)
    plt.ylim(0, 1)
    plt.legend(loc='upper left')

    return (plt.gca(), )

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=1000)
arguments.add_argument('--alpha', type=float, default=0.1)
args = arguments.parse_args()

model = TemporalDifference(args.states, args.episodes, args.alpha)
ani = FuncAnimation(plt.gcf(),
                    func,
                    frames=enumerate(model),
                    init_func=lambda: (plt.gca(), ),
                    interval=50,
                    save_count=args.episodes)
ani.save('example-6.2.mp4')

from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def func(frame, limit):
    (i, (_, data)) = frame

    plt.clf()
    ax = sns.heatmap(data, vmin=-limit, vmax=limit)
    ax.set_title('Iteration: {}'.format(i))
    ax.invert_yaxis()

    return [ ax ]
        
arguments = ArgumentParser()
arguments.add_argument('--data', type=Path)
arguments.add_argument('--limit', type=int)
args = arguments.parse_args()

data = np.load(args.data)
ani = FuncAnimation(plt.gcf(),
                    func,
                    frames=enumerate(data.items()),
                    fargs=(args.limit, ),
                    interval=1000)
ani.save(args.data.with_suffix('.mp4'))

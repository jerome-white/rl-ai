from pathlib import Path
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def heatmaps(data, minmax):
    iterable = iter(data)
    
    def wrapper(_):
        i = next(iterable)
        (_, title) = i.split('_')

        plt.clf()
        ax = sns.heatmap(data[i], vmin=-minmax, vmax=minmax)
        ax.set_title('Iteration: ' + title)
        ax.invert_yaxis()

    return wrapper
        
arguments = ArgumentParser()
arguments.add_argument('--data', type=Path)
arguments.add_argument('--config', type=Path)
args = arguments.parse_args()

config = ConfigParser()
config.read(args.config)

data = np.load(args.data)
artists = heatmaps(data, int(config['system']['movable']))
frames = len(data.keys()) - 1

ani = FuncAnimation(plt.gcf(),
                    artists,
                    frames=frames,
                    interval=1000,
                    repeat=True)
ani.save(args.data.with_suffix('.mp4'))

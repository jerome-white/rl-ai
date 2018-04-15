import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

arguments = ArgumentParser()
arguments.add_argument('--output', type=Path)
args = arguments.parse_args()

df = (pd
      .read_csv(sys.stdin, index_col=False)
      .groupby(['epsilon', 'play'])
      .mean()
      .unstack(level=0))

plt.style.use('ggplot')
(figure, axes) = plt.subplots(nrows=2, sharex=False)
for (ax, factor) in zip(axes, ('reward', 'optimal')):
    df[factor].plot(ax=ax)
plt.savefig(str(args.output), bbox_inches='tight')

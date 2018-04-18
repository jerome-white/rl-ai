import sys
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

arguments = ArgumentParser()
arguments.add_argument('--output', type=Path)
arguments.add_argument('--factor')
args = arguments.parse_args()

assert(args.factor == 'epsilon' or args.factor == 'temperature')

df = (pd
      .read_csv(sys.stdin, index_col=False)
      .groupby([args.factor, 'play'])
      .mean()
      .unstack(level=0))

plt.style.use('ggplot')

(figure, axes) = plt.subplots(nrows=2, sharex=False)
figure.set_size_inches(figure.get_size_inches() * (1.5, 2.25))

for (ax, factor) in zip(axes, ('reward', 'optimal')):
    df[factor].plot(ax=ax)
plt.savefig(str(args.output), bbox_inches='tight')

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns

from walk import TemporalDifference

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=1000)
arguments.add_argument('--alpha', type=float, default=0.1)
arguments.add_argument('--samples', action='append')
args = arguments.parse_args()

if args.samples:
    samples = args.samples
else:
    samples = set([0] + np.logspace(0, 3, 4, dtype=int).tolist())

model = TemporalDifference(args.states, args.episodes, args.alpha)
df = pd.DataFrame.from_dict(dict(enumerate(model)), orient='index')
df = df[df.index.isin(samples)]

ax = sns.lineplot(data=df)
ax.get_figure().savefig('example-6.2.png')

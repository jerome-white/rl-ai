import sys

import pandas as pd
import matplotlib.pyplot as plt

df = (pd
      .read_csv(sys.stdin, index_col=False)
      .groupby(['epsilon', 'play'])
      .mean()
      .unstack(level=0))

plt.style.use('ggplot')
(figure, axes) = plt.subplots(nrows=2, sharex=False)
for (ax, factor) in zip(axes, ('reward', 'optimal')):
    df[factor].plot(ax=ax)
plt.savefig('2.1.png', bbox_inches='tight')

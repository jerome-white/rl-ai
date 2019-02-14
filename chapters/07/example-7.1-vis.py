import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = (pd
      .read_csv(sys.stdin)
      .sort_values(by=[
          'online',
          'alpha',
          'steps',
      ]))

for (i, data) in df.groupby('online'):
    plt.clf()
    sns.lineplot(x='alpha',
                 y='rmse',
                 hue='steps',
                 data=data,
                 sort=False)
    plt.savefig(str(i) + '.png')

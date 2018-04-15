import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(sys.stdin, index_col=False)
df = df.melt(id_vars=['epsilon', 'bandit', 'play'],
             value_vars=['reward', 'optimal'])

g = sns.FacetGrid(data=df, row='variable', hue='epsilon', sharey=False)
g.map(sns.pointplot,
      'step',
      'value',
      order=sorted(df['step'].unique()),
      ci=None)
g.savefig('2.1.png', bbox_inches='tight')

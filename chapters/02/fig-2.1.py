import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(sys.stdin, index_col=False)
df = df.melt(id_vars=['epsilon', 'bandit', 'play'],
             value_vars=['reward', 'optimal'])

g = sns.FacetGrid(data=df, row='variable', hue='epsilon', sharey=False)
g.map(sns.pointplot,
      'play',
      'value',
      order=sorted(df['play'].unique()))
g.savefig('2.1.png', bbox_inches='tight')

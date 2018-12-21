import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(sys.stdin)

sns.lineplot(x='steps',
             y='episodes',
             hue='order',
             estimator=None,
             data=df)

plt.grid(True)
plt.legend().remove()
plt.savefig('example-6.11.png')

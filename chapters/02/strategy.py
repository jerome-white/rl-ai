import numpy as np

class SelectionStrategy:
    def choose(self, arms):
        raise NotImplementedError()

class Explore(SelectionStrategy):
    def choose(self, arms):
        return np.random.choice(arms)

class SoftMax(SelectionStrategy):
    def __init__(self, temperature=1):
        assert(temperature != 0)

        self.temperature = temperature

    def choose(self, arms):
        p = np.array([ np.exp(float(x) / self.temperature) for x in arms ])
        p /= np.sum(p)
        return np.random.choice(arms, p=p)

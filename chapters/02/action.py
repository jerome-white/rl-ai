import numpy as np

class Arm:
    def __init__(self, name, alpha=0, reward=None):
        self.name = name
        self.alpha = alpha
        self.reward = np.random.normal() if reward is None else reward

        self.pulls = 0
        self.estimate = float(0)

    def __eq__(self, other):
        return self.name == other.name

    def __float__(self):
        return self.estimate

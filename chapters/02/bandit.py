import collections as cl

import pandas as pd
import numpy as np

class Action:
    def __init__(self, name, step=None):
        self.name = name
        self.step = step

        self.reward = np.random.normal()
        self.pulls = 0
        self.estimate = 0

    def __eq__(self, other):
        return self.name == other.name

    def __float__(self):
        return self.estimate

    # def __gt__(self, other):
    #     return float(self) > float(other)

    def execute(self):
        reward = np.random.normal(self.reward)

        alpha = 1 / (self.pulls + 1) if self.step is None else self.alpha
        self.estimate += alpha * (value - self.estimate)
        self.pulls += 1

        return reward

class Bandit:
    def __init__(self, arms, epsilon=0, temperature=0):
        self.epsilon = epsilon

        if temperature:
            self.softmax = lambda x: np.exp(float(x) / temperature)
        else:
            self.softmax = None

        self.actions = [ Action(x) for x in range(arms) ]

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.binomial(1, self.epsilon):
            # explore
            if self.softmax:
                p = np.array([ self.softmax(x) for x in self.actions ])
                p /= np.sum(p)
            else:
                p = None
            action = np.random.choice(self.actions, p=p)
        else:
            # exploit
            estimates = list(map(float, self.actions))
            largest = np.argwhere(estimates == np.max(estimates)).flatten()
            action = self.actions[np.random.choice(largest)]

        return action

    def pull(self, action):
        return action.execute()

    def isoptimal(self, action):
        return action == max(self.actions, key=lambda x: x.reward)

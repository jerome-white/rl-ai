import collections as cl

import pandas as pd
import numpy as np

class SampleAverage:
    def __init__(self):
        self.i = 0
        self.value = float(0)

    def __float__(self):
        return self.value

    def update(self, value):
        i = self.i + 1
        self.value = (self.i * self.value + value) / i
        self.i = i

class Action:
    def __init__(self, name):
        self.name = name
        self.reward = np.random.normal()
        self.estimate = SampleAverage()

    def __eq__(self, other):
        return self.name == other.name

    def execute(self):
        reward = np.random.normal(self.reward)
        self.estimate.update(reward)

        return reward

class Bandit:
    def __init__(self, arms, epsilon=0, temperature=0):
        self.arms = arms
        self.epsilon = epsilon

        if temperature:
            self.softmax = lambda x: np.exp(float(x.estimate) / temperature)
        else:
            self.softmax = None

        self.actions = [ Action(x) for x in range(self.arms) ]

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
            estimates = [ float(x.estimate) for x in self.actions ]
            largest = np.argwhere(estimates == np.max(estimates)).flatten()
            action = self.actions[np.random.choice(largest)]

        return action

    def pull(self, action):
        return action.execute()

    def isoptimal(self, action):
        return action == max(self.actions, key=lambda x: x.reward)

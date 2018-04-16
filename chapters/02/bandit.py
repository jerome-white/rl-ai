import numpy as np
import collections as cl

Result = cl.namedtuple('Result', 'epsilon, bandit, play, reward, optimal')

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
        self.selected = 0
        self.reward = np.random.randn()
        self.estimate = SampleAverage()

    def __eq__(self, other):
        return self.name == other.name

class Bandit:
    def __init__(self, arms, epsilon=0):
        self.arms = arms
        self.epsilon = epsilon

        # self.plays = 0
        # self.points = 0
        self.actions = [ Action(x) for x in range(self.arms) ]

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.binomial(1, self.epsilon):
            # explore
            action = np.random.choice(self.actions)
        else:
            # exploit
            action = max(self.actions, key=lambda x: float(x.estimate))
        action.selected += 1

        return action

    def pull(self, action):
        reward = action.reward
        action.estimate.update(reward)

        return reward

    def isoptimal(self, action):
        return action == max(self.actions, key=lambda x: x.reward)

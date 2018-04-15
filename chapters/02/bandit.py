import numpy as np
import operator as op
import collections as cl

Result = cl.namedtuple('Result', 'epsilon, bandit, play, reward, optimal')

class Action:
    def __init__(self, name):
        self.name = name
        self.reward = np.random.randn()
        self.estimate = 0
        self.chosen = 0

    def __eq__(self, other):
        return self.name == other.name

    def activate(self):
        self.chosen += 1
        self.estimate += 1 / self.chosen * (self.reward - self.estimate)

class Bandit:
    def __init__(self, arms, epsilon=0):
        self.arms = arms
        self.epsilon = epsilon

        self.plays = 0
        self.points = 0
        self.actions = [ Action(x) for x in range(self.arms) ]

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.binomial(1, self.epsilon):
            action = np.random.choice(self.actions) # explore
        else:
            action = max(self.actions, key=op.attrgetter('estimate')) # exploit

        return action

    def do(self, action):
        plays = self.plays + 1
        self.points = (self.plays * self.points + action.reward) / plays
        self.plays = plays

        action.activate()

    def isoptimal(self, action):
        return action == max(self.actions, key=op.attrgetter('reward'))

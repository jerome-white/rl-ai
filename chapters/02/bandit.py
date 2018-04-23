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

    def execute(self):
        reward = np.random.normal(self.reward)

        alpha = self.alpha if self.alpha else 1 / (self.pulls + 1)
        self.estimate += alpha * (reward - self.estimate)
        self.pulls += 1

        return reward

class Bandit:
    def __init__(self, arms, explorer, epsilon=0):
        self.arms = arms
        self.epsilon = epsilon
        self.explorer = explorer

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.binomial(1, self.epsilon):
            # explore
            action = self.explorer.choose(self.arms)
        else:
            # exploit
            estimates = list(map(float, self.arms))
            largest = np.argwhere(estimates == np.max(estimates)).flatten()
            action = self.arms[np.random.choice(largest)]

        return action

    def pull(self, arm):
        return arm.execute()

    def isoptimal(self, action):
        return action == max(self.arms, key=lambda x: x.reward)

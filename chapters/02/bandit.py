import numpy as np

from strategy import SelectionStrategy, SoftMax

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

    def isoptimal(self, action):
        return action == max(self.arms, key=lambda x: x.reward)

    def pull(self, arm):
        reward = np.random.normal(arm.reward)
        self.update(arm, reward)

        return reward

    def update(self, arm, reward):
        raise NotImplementedError()

class ActionRewardBandit(Bandit):
    def update(self, arm, reward):
        alpha = arm.alpha if arm.alpha else 1 / (arm.pulls + 1)
        arm.estimate += alpha * (reward - arm.estimate)
        arm.pulls += 1

class ReinforcementBandit(Bandit):
    def __init__(self, arms, alpha, beta, reference=0):
        assert(0 < alpha <= 1)

        super().__init__(arms, SoftMax(), 1)

        self.alpha = alpha
        self.beta = beta
        self.reference = reference

    def update(self, arm, reward):
        reference = reward - self.reference
        arm.estimate += self.beta * reference
        self.reference += self.alpha * reference

class PursuitBandit(Bandit):
    def __init__(self, arms, beta):
        super().__init__(arms, SelectionStrategy())

        self.beta = beta

    def update(self, arm, reward):
        for a in self.arms:
            factor = int(a == arm)
            a.estimate += self.beta * (factor - a.estimate)

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
        self._pull(arm, reward)

        return reward

    def _pull(self, arm, reward):
        raise NotImplementedError()

class ActionRewardBandit(Bandit):
    def _pull(self, arm, reward):
        alpha = arm.alpha if arm.alpha else 1 / (arm.pulls + 1)
        arm.estimate += alpha * (reward - arm.estimate)
        arm.pulls += 1

class ReinforcementBandit(Bandit):
    def __init__(self, arms, beta, alpha, reference=0):
        super().__init__(arms, SoftMax(1), 1)

        assert(0 < alpha <= 1)

        self.alpha = alpha
        self.beta = beta
        self.reference = reference

    def _pull(self, arm, reward):
        reference = reward - self.reference
        arm.estimate += self.beta * reference
        self.reference += self.alpha * reference

class PursuitBandit(Bandit):
    def __init__(self, arms, beta):
        super().__init__(arms, None)

        self.beta = beta

    def _pull(self, arm, reward):
        for a in self.arms:
            a.estimate += self.beta * (int(a == arm) - a.estimate)

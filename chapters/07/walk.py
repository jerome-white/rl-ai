import random
import functools as ft
import collections as cl

import numpy as np

Transition = cl.namedtuple('Transition', 'state, action, reward')

@ft.lru_cache(maxsize=2**10)
def power(base, exponent):
    return base ** exponent

def walk(end, start=0):
    reward = 0
    state = end // 2

    while not reward:
        action = random.choice((-1, 1))
        state_ = state + action
        assert(start <= state_ <= end)

        if state_ == start:
            reward = -1
        elif state_ == end:
            reward = 1

        yield Transition(state, action, reward)
        state = state_

class TemporalDifference:
    def __init__(self, states, episodes, alpha, gamma, n=None):
        self.V = np.zeros(states)
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.n = n

        self.step = 0

    def __iter__(self):
        assert(self.episodes >= 0)
        return self

    def __next__(self):
        if self.step:
            if self.step > self.episodes:
                raise StopIteration()
            self.update()
        self.step += 1

        return self.V

    # Corrected n-step truncated return (pg. 165)
    def R(self, window):
        reward = 0
        last = len(window) if self.n is None else self.n
        last -= 1

        for (i, t) in enumerate(window):
            r = t.reward if i < last else self.V[t.state]
            reward += power(self.gamma, i) * r

        return reward

    # Page 166: \Delta V_{t}(S_{t})
    def delta(self, window, state):
        return self.alpha * (self.R(window) - self.V[state])

    # Page 166: V(s) + \sum_{t=0}^{T-1}\Delta V_{t}(s)
    def update(self):
        raise NotImplementedError()

    @staticmethod
    def rmse(V):
        count = len(V)
        optimal = np.arange(-count, count, step=2) / count
        distance = V - optimal

        return np.sqrt(np.sum(np.power(distance, 2)) / len(V))

class OnlineUpdate(TemporalDifference):
    def update(self):
        step = walk(len(self.V))
        window = cl.deque(maxlen=self.n)

        while True:
            try:
                window.append(next(step))
                if window.maxlen and len(window) < window.maxlen:
                    continue
            except StopIteration:
                window.popleft()
                if not window:
                    break
            state = window[0].state
            self.V[state] += self.delta(window, state)

class OfflineUpdate(TemporalDifference):
    def update(self):
        return

import random
import functools as ft
import collections as cl

import numpy as np

Transition = cl.namedtuple('Transition', 'state, action, reward')

@ft.lru_cache(maxsize=2**10)
def power(base, exponent):
    return base ** exponent

def walk(states, initial=0):
    reward = 0
    state = states // 2

    while not reward:
        action = random.choice((-1, 1))
        state_ = state + action

        if state_ <= initial:
            reward = -1
        elif state_ >= states:
            reward = 1

        yield Transition(state, action, reward)
        state = state_

class TemporalDifference:
    def __init__(self, states, episodes, alpha, gamma, n=None):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.n = n

        self.step = 0
        self.V = np.zeros(states + 2)

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

    # Corrected n-step truncated return (p6. 165)
    def R(self, window):
        last = len(window) - 1
        for (i, t) in enumerate(window):
            reward = t.reward if i < last else self.V[t.state]
            yield power(self.gamma, i) * reward

    # Page 166: \Delta V_{t}(S_{t})
    def delta(self, window, state):
        reward = sum(self.R(window))
        return self.alpha * (reward - self.V[state])

    # Page 166: V(s) + \sum_{t=0}^{T-1}\Delta V_{t}(s)
    def update(self):
        raise NotImplementedError()

class OnlineUpdate(TemporalDifference):
    def update(self):
        window = cl.deque(maxlen=self.n)

        for trans in walk(len(self.V)):
            window.append(trans)
            if not window.maxlen or len(window) == window.maxlen:
                s = trans.state
                self.V[s] += self.delta(window, s)

class OfflineUpdate(TemporalDifference):
    def update(self):
        return

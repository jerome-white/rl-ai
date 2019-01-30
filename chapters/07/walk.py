import random
import itertools as it
import functools as ft
import collections as cl

import numpy as np

Transition = cl.namedtuple('Transition', 'state, action, reward')

@ft.lru_cache(maxsize=2**10)
def power(base, exponent):
    return base ** exponent

def walk(states, initial=0):
    state = states // 2

    while True:
        action = random.choice((-1, 1))
        state_ = state + action

        if 0 < state_ < states:
            yield Transition(state, action, 0)
            state = state_
        else:
            reward = 1 if state_ else -1
            return Transition(state, action, reward)

class Model:
    def __init__(self, states, episodes, alpha):
        self.episodes = episodes
        self.alpha = alpha

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

    def update(self):
        raise NotImplementedError()

class TemporalDifference(Model):
    def __init__(self, states, episodes, alpha, gamma, n=None):
        super().__init__(states, episodes, alpha)

        self.gamma = gamma
        self.n = n

    def R(self, window, time, state):
        start = time - 1
        if start >= 0:
            stop = len(window) - 1
            for (i, s) in enumerate(it.islice(window, start, stop)):
                yield power(self.gamma, i) * s.reward
        yield power(self.gamma, time) * self.V[state]

    def delta(self, window):
        for (i, s) in enumerate(window):
            reward = sum(self.R(window, i, s.state))
            yield self.alpha * (reward - self.V[s.state])

    def update(self):
        window = cl.deque(maxlen=self.n)

        for trans in walk(len(self.V)):
            if not self.n or window.maxlen == self.n:
                self.V[trans.state] += sum(self.delta(window))
            window.append(trans)

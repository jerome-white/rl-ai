import random
import operator as op
import itertools as it
import functools as ft
import collections as cl

import numpy as np

Transition = cl.namedtuple('Transition', 'state, action, reward')

@ft.lru_cache(maxsize=2**10)
def power(base, exponent):
    return base ** exponent

def walk(states, pos=None, terminal=np.inf):
    assert(not any([ f(terminal) in states for f in (op.neg, op.pos) ]))

    n = len(states)

    if pos is None:
        pos = n // 2

    while 0 <= pos < n:
        action = random.choice((-1, 1))
        pos_ = pos + action
        if pos_ < n:
            reward = 0
        elif pos_ == n:
            reward = 1
        else:
            reward = -1

        yield Transition(states[pos], action, reward)

        pos = pos_

    yield Transition(np.sign(pos) * terminal, 0, np.nan)

class Model:
    def __init__(self, states, episodes, n, alpha):
        self.episodes = episodes
        self.alpha = alpha

        self.step = 0
        self.states = []
        self.V = cl.defaultdict(float)

        for i in range(states):
            s = chr(ord('A') + i)
            self.states.append(s)
            self.V[s] = 0.5

    def __iter__(self):
        assert(self.episodes >= 0)

        return self

    def __next__(self):
        if self.step:
            if self.step > self.episodes:
                raise StopIteration()
            self.update()
        self.step += 1

        return { x: self.V[x] for x in self.states }

    def update(self):
        raise NotImplementedError()

class NStep(Model):
    def __init__(self, states, episodes, alpha, n=None):
        super().__init__(states, episodes, alpha)

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
            reward = sum(self.R(window, i, s))
            yield self.alpha * (reward - self.V[s.state])

    def update(self):
        reward = 0
        window = cl.deque(maxlen=self.n)

        for state in walk(self.states):
            if not self.n or window.maxlen == self.n:
                self.V[state] += sum(self.delta(window))
            window.append(state)

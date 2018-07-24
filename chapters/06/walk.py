import random
import operator as op
import collections as cl

import numpy as np

Transition = cl.namedtuple('Transition', 'state, action, reward')

def walk(states, pos=None, terminal=np.inf):
    assert(not any([ f(terminal) in states for f in (op.neg, op.pos) ]))

    n = len(states)

    if pos is None:
        pos = n // 2

    while 0 <= pos < n:
        action = random.choice((-1, 1))
        pos_ = pos + action
        reward = 0 if pos_ < n else 1

        yield Transition(states[pos], action, reward)

        pos = pos_

    yield Transition(np.sign(pos) * terminal, 0, np.nan)

class Model:
    def __init__(self, states, episodes, alpha, gamma):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma

        self.i = 0
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
        if self.i:
            if self.i > self.episodes:
                raise StopIteration()
            self.step()
        self.i += 1

        return self.V

    def step(self):
        raise NotImplementedError()

class TemporalDifference(Model):
    def step(self):
        s = None

        for s_ in walk(self.states):
            if s is not None:
                v = s.reward + self.gamma * self.V[s_.state] - self.V[s.state]
                self.V[s.state] += self.alpha * v
            s = s_

class MonteCarlo(Model):
    def step(self):
        episode = list(walk(self.states))
        episode.pop()

        reward = episode[-1].reward

        for s in reversed(episode):
            v = self.gamma * reward - self.V[s.state]
            self.V[s.state] += self.alpha * v

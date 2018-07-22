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

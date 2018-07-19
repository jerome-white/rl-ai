import random
import collections as cl

Transition = cl.namedtuple('Transition', 'state, action, reward')

def walk(states=6, start=None):
    if start is None:
        start = states // 2

    while 0 < start < states:
        action = random.choice((-1, 1))
        start_ = start + action
        reward = 1 if start_ == states else 0

        yield Transition(start, action, reward)

        start = start_

import random
import collections as cl

Transition = cl.namedtuple('Transition', 'state, action, reward')

def walk(states=6, start=None):
    if start is None:
        start = states // 2

    while True:
        action = random.choice((-1, 1))
        start_ = start + action
        reward = 1 if start_ == states else 0

        state = chr(ord('A') + start - 1)

        yield Transition(state, action, reward)

        if not (0 < start < states):
            break
        start = start_

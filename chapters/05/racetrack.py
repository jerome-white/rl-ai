import csv
import random
import operator as op
import itertools as it
import collections as cl

import numpy as np

State = cl.namedtuple('State', 'position, velocity')
Transition = cl.namedtuple('Transition', 'state, action, reward')
_Vector = cl.namedtuple('_Vector', 'x, y')

def actions():
    values = it.product(range(-1, 2), repeat=len(Vector._fields))
    yield from it.starmap(Vector, values)

class Vector(_Vector):
    def __new__(cls, x, y):
        return super(Vector, cls).__new__(cls, x, y)

    def __add__(self, other):
        return type(self)(*it.starmap(op.add, zip(self, other)))

    def __gt__(self, other):
        return self.x < other.x or self.y > other.y

    def __bool__(self):
        return any(self)

    def __str__(self):
        return ','.join(map(str, self))

    def __repr__(self):
        return str(self)

    def advance(self, other):
        return type(self)(self.x - other.x, self.y + other.y)

    def clip(self, at=4):
        return type(self)(*map(lambda x: np.clip(x, 0, at), self))

class Track:
    def __init__(self, track, start='s', finish='f', out='.'):
        self.track = []
        self.start = set()
        self.finish = set()

        with track.open() as fp:
            reader = csv.reader(fp)
            for line in reader:
                row = []
                i = reader.line_num - 1
                for (j, cell) in enumerate(line):
                    inbounds = cell != out
                    row.append(inbounds)
                    if inbounds and (cell == start or cell == finish):
                        kind = self.start if cell == start else self.finish
                        kind.add(Vector(i, j))
                self.track.append(row)

    def __getitem__(self, key):
        try:
            if all([ x >= 0 for x in key ]):
                return self.track[key.x][key.y]
        except IndexError:
            pass

        return False

    def navigate(self, position, velocity):
        iterable = it.product(*map(lambda x: range(x + 1), velocity))
        for i in it.starmap(Vector, filter(lambda x: any(x), iterable)):
            pos = position.advance(i)
            yield (pos, self[pos])

class Race:
    def __init__(self, state, track, reward=-1, penalty=4):
        self.state = state
        self.track = track
        self.reward = reward
        self.penalty = penalty

    def __iter__(self):
        assert(self.state.position in self.track.start)
        return self

    def __next__(self):
        if self.state.position in self.track.finish:
            raise StopIteration()

        #
        # Select and take an action
        #
        while True:
            action = random.choice(list(actions()))
            velocity = (self.state.velocity + action).clip()
            if velocity:
                break

        #
        # Calculate the destination based on this action
        #
        displacement = self.displace(velocity)
        route = list(self.track.navigate(self.state.position, displacement))
        (position, inbounds) = max(route, key=op.itemgetter(0)) # final dest.

        reward = self.reward

        # If any of the sectors along the route are out of bounds, the
        # car drove off the road to get to this destination.
        if not all(map(op.itemgetter(1), route)):
            reward -= self.penalty

            # If the final destination is out of bounds, choose a new
            # position that's one-closer to the finish.
            if not inbounds:
                filt = lambda x: [ a for (a, b) in x if b ]
                good = filt(route)
                if not good:
                    step = Vector(1, 1)
                    good = filt(self.track.navigate(self.state.position, step))
                    assert(good)
                position = max(good)

        transition = Transition(self.state, action, reward)
        self.state = State(position, velocity)

        return transition

    def displace(self, velocity):
        raise NotImplementedError()

class FlatRace(Race):
    def displace(self, velocity):
        return velocity

class DownhillRace(Race):
    def __init__(self, state, track, reward=-1, penalty=4, gradient=1):
        super().__init__(state, track, reward, penalty)

        self.step = 0
        self.gradient = gradient + 1

    def displace(self, velocity):
        if self.step % 2:
            args = []
            for _ in range(len(Vector._fields)):
                gradient = random.randrange(self.gradient)
                args.append(gradient)
            velocity += Vector(*args)
        self.step += 1

        return velocity

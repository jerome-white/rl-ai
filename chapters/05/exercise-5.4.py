import csv
import random
import logging
import operator as op
import itertools as it
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(process)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

State = cl.namedtuple('State', 'position, velocity')
Transition = cl.namedtuple('Transition', 'state, action, reward')
_Vector = cl.namedtuple('_Vector', 'x, y')

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
            for (i, line) in enumerate(reader):
                i = reader.line_num - 1
                row = []
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
        (position, inbounds) = max(route, key=op.itemgetter(0))

        if not inbounds:
            filt = lambda x: [ a for (a, b) in x if b ]

            elegible = filt(route)
            if not elegible:
                step = Vector(1, 1)
                elegible = filt(self.track.navigate(self.state.position, step))
                assert(elegible)
            position = max(elegible)

            reward = -5
        elif not all(map(op.itemgetter(1), route)):
            reward = -5
        else:
            reward = -1

        transition = Transition(self.state, action, reward)
        self.state = State(position, velocity)

        return transition

    def displace(self, velocity):
        raise NotImplementedError()

class FlatRace(Race):
    def displace(self, velocity):
        return velocity

class DownhillRace(Race):
    def __init__(self, track, start='s', finish='f', out='.', gradient=1):
        super().__init__(track, start, finish, out)

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

def actions():
    values = it.product(range(-1, 2), repeat=len(Vector._fields))
    yield from it.starmap(Vector, values)

def func(queue, games, track, setting):
    while True:
        position = queue.get()
        start = State(position, Vector(0, 0))

        returns = cl.defaultdict(list)
        values = cl.defaultdict(float)
        policy = {}

        for i in range(games):
            logging.info('{0} {1}'.format(start, i))

            #
            # generate epsiode and calculate returns
            #
            race = setting(start, track)
            states = []

            for transition in race:
                logging.debug(transition)

                key = (transition.state, transition.action)
                returns[key].append(transition.reward)
                values[key] = np.mean(returns[key])

                states.append(transition.state)

            #
            # calculate optimal policies
            #
            for s in states:
                vals = cl.defaultdict(list)
                for a in actions():
                    key = values[(s, a)]
                    vals[key].append(a)
                best = max(vals.keys())
                policy[s] = random.choice(vals[best])

        queue.task_done()

arguments = ArgumentParser()
arguments.add_argument('--track', type=Path)
arguments.add_argument('--games', type=int)
arguments.add_argument('--downhill', action='store_true')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

queue = mp.JoinableQueue()
track = Track(args.track)
setting = DownhillRace if args.downhill else FlatRace

with mp.Pool(args.workers, func, (queue, args.games, track, setting)):
    for i in track.start:
        queue.put(i)
    queue.join()

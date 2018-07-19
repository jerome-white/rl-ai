import random
import logging
import collections as cl
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

import racetrack as rt
from racetrack import State, Vector, Track, FlatRace, DownhillRace

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(process)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

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
                for a in rt.actions():
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

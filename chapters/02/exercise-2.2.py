import sys
import csv
import logging
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

from bandit import Bandit, Arms
from sstrat import Explore, SoftMax

result = [
    'bandit',
    'epsilon',
    'play',
    'reward',
    'optimal',
]
Result = cl.namedtuple('Result', result)

def run(incoming, outgoing, pulls, arms):
    while True:
        (bid, epsilon, temperature) = incoming.get()
        logging.info('{0}: {1}'.format(bid, bargs))

        arms = [ Arm(x) for x in range(arms) ]
        explore = SoftMax(temperature) if temperature else Explore()
        bandit = Bandit(arms, Explore(), epsilon)
        
        for (play, arm) in enumerate(it.islice(bandit, 0, pulls)):
            reward = bandit.pull(arm)
            optimal = int(bandit.isoptimal(arm))
            result = Result(bid, epsilon, temperature, play, reward, optimal)
            
            outgoing.put(result)
        outgoing.put(None)

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--bandits', type=int)
arguments.add_argument('--arms', type=int)
arguments.add_argument('--pulls', type=int)
arguments.add_argument('--epsilon', type=float, action='append')
arguments.add_argument('--temperature', type=float, action='append')
arguments.add_argument('--workers', type=int, default=mp.cpu_count())
args = arguments.parse_args()

incoming = mp.Queue()
outgoing = mp.Queue()

initargs = (outgoing, incoming, args.pulls, args.arms)

with mp.Pool(args.workers, run, initargs) as pool:
    jobs = 0
    iterable = it.product(range(args.bandits),
                          args.epsilon or [ 0 ],
                          args.temperature or [ 0 ])
    for i in iterable:
        outgoing.put(i)
        jobs += 1

    writer = csv.DictWriter(sys.stdout, fieldnames=Result._fields)
    writer.writeheader()

    while jobs:
        play = incoming.get()
        if play is None:
            jobs -= 1
        else:
            writer.writerow(play._asdict())

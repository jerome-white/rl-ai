import sys
import csv
import logging
import inspect
import itertools as it
import collections as cl
import multiprocessing as mp
from argparse import ArgumentParser

from bandit import Bandit

bsig = inspect.signature(Bandit)
BanditArgs = cl.namedtuple('BanditArgs', bsig.parameters.keys())
fields = BanditArgs._fields + ('bandit', 'play', 'reward', 'optimal')
Result = cl.namedtuple('Result', fields)

def run(incoming, outgoing, pulls):
    while True:
        (bid, bargs) = incoming.get()
        logging.info('{0}: {1}'.format(bid, bargs))

        bandit = Bandit(*bargs)
        for (play, arm) in enumerate(it.islice(bandit, 0, pulls)):
            reward = bandit.pull(arm)
            optimal = int(bandit.isoptimal(arm))
            result = it.chain(bargs, (bid, play, reward, optimal))

            outgoing.put(Result._make(result))
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

initargs = (outgoing, incoming, args.pulls)

with mp.Pool(args.workers, run, initargs) as pool:
    jobs = 0
    dimensions = [
        range(args.bandits),
        [ args.arms ],
        [ 0 ] if not args.epsilon else args.epsilon,
        [ 0 ] if not args.temperature else args.temperature,
    ]
    
    for (i, *args) in it.product(*dimensions):
        outgoing.put((i, BanditArgs._make(args)))
        jobs += 1

    writer = csv.DictWriter(sys.stdout, fieldnames=Result._fields)
    writer.writeheader()

    while jobs:
        play = incoming.get()
        if play is None:
            jobs -= 1
        else:
            writer.writerow(play._asdict())

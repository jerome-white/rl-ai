import sys
import csv
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

from walk import TemporalDifference

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

arguments = ArgumentParser()
arguments.add_argument('--states', type=int, default=5)
arguments.add_argument('--episodes', type=int, default=1)
arguments.add_argument('--alpha', type=float, default=1)
arguments.add_argument('--gamma', type=float, default=1)
args = arguments.parse_args()

model = TemporalDifference(args.states, args.episodes, args.alpha, args.gamma)

writer = csv.DictWriter(sys.stdout,
                        fieldnames=model.states,
                        extrasaction='ignore')
writer.writeheader()
writer.writerows(model)

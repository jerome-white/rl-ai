import random
import logging
import itertools as it
import collections as cl
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG,
                    format='[ %(asctime)s ] %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')

Card = cl.namedtuple('Card', 'suit, value')
State = cl.namedtuple('State', 'player, dealer, ace')

class Average:
    def __init__(self):
        self.average = 0
        self.n = 1

    def add(self, value):
        self.average += (value - self.average) / self.n
        self.n += 1

    def __float__(self):
        return self.average

class Deck(list):
    def __init__(self):
        for i in range(4):
            # ace
            self.append(Card(i, None))

            # number cards
            for j in range(2, 14):
                value = min(j, 10) # jack, queen, king are 10
                self.append(Card(i, value))

    def __next__(self):
        return random.choice(self)

class Policy:
    def __init__(self):
        self.ace = False
        self.cards = []
        self.value = 0

    def __int__(self):
        return self.value

    def __bool__(self):
        return int(self) <= 21 and not self.stick(int(self))

    def __str__(self):
        return '[{0}]'.format(','.join([ str(x.value) for x in self.cards ]))

    def deal(self, card):
        self.cards.append(card)

        try:
            self.value += card.value
        except TypeError:
            if self.value + 11 <= 21:
                self.value += 11
                self.ace = True
            else:
                self.value += 1

        if self.value > 21:
            raise OverflowError()

    def isnatural(self):
        return len(self.cards) == 2 and int(self) == 21

    def stick(self, value):
        raise NotImplementedError()

class Dealer(Policy):
    def stick(self, value):
        return value >= 17

class Player(Policy):
    def stick(self, value):
        return value == 20 or value == 21

def play():
    episode = []

    deck = Deck()
    dealer = Dealer()
    player = Player()

    for p in (player, dealer):
        for _ in range(2):
            p.deal(next(deck))

    face = dealer.cards[0].value
    episode = [ State(int(player), face, player.ace) ]

    if player.isnatural() and not dealer.isnatural():
        return (episode, 1)

    for (i, p) in enumerate((player, dealer)):
        while p:
            bust = False
            try:
                p.deal(next(deck))
            except OverflowError:
                bust = True

            if not i:
                episode.append(State(int(p), face, p.ace))

            if bust:
                reward = 1 if i else -1
                return (episode, reward)

    (p, d) = [ int(x) for x in (player, dealer) ]
    reward = ((p > d) - (p < d)) # https://stackoverflow.com/a/11215908

    return (episode, reward)

arguments = ArgumentParser()
arguments.add_argument('--games', type=int)
arguments.add_argument('--with-ace', action='store_true')
args = arguments.parse_args()

values = cl.defaultdict(Average)
for _ in range(args.games):
    #
    # generate epsiode
    #
    (episode, reward) = play()
    logging.info('{0} {1}'.format(','.join(map(str, episode)), reward))

    #
    # calculate returns
    #
    for state in episode:
        values[state].add(reward)

V = np.zeros((21 - 12 + 1, 10 - 2 + 1))
for (k, v) in values.items():
    if args.with_ace ^ v.ace:
        V[k.player, k.dealer] = float(v)

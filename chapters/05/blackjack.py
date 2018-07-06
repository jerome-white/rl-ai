import random
import collections as cl

import numpy as np

Card = cl.namedtuple('Card', 'suit, value')
State = cl.namedtuple('State', 'player, dealer, ace')

def fairmax(Q, state):
    vals = [ Q[(state, x)] for x in range(2) ]
    best = np.argwhere(vals == np.max(vals))

    return np.random.choice(best.flatten())

class Deck(list):
    def __init__(self):
        for i in range(4):
            for j in range(1, 14):
                card = Card(i, min(j, 10))
                self.append(card)

    def __next__(self):
        return random.choice(self)

class Policy:
    def __init__(self, value=0, cards=0, ace=False):
        self.value = value
        self.cards = cards
        self.ace = ace

    def __int__(self):
        return self.value

    def __bool__(self):
        return self.cards == 2 and self.value == 21

    def __str__(self):
        return '{0:2d}/{1}/{2:d}'.format(self.value, self.cards, self.ace)

    def deal(self, card):
        self.cards += 1

        self.value += card.value
        if card.value == 1 and self.value + 10 <= 21:
            self.use_ace()

        while self.value > 21:
            if self.ace:
                self.value -= 10
                self.ace = False
            else:
                raise OverflowError()

    def use_ace(self):
        self.value += 10
        self.ace = True

    def tostate(self, facecard):
        return State(self.value, facecard, self.ace)

    def hit(self, facecard):
        raise NotImplementedError()

class Dealer(Policy):
    def hit(self, facecard):
        return self.value < 17

    def deal(self, card):
        if self.cards == 1 and self.value == 1 and card.value != 1:
            self.use_ace()

        super().deal(card)

class Player(Policy):
    def hit(self, facecard):
        return self.value < 20

class Blackjack:
    def __init__(self, state=None, player=Player):
        self.deck = Deck()

        if state is None:
            self.table = (player(), Dealer())

            for (i, p) in enumerate(self.table):
                for _ in range(2):
                    card = next(self.deck)
                    p.deal(card)

                    # Is this the dealers first card?
                    if i and p.cards == 1:
                        self.face = card.value
        else:
            self.table = (
                player(value=state.player, cards=2, ace=state.ace),
                Dealer(state.dealer, 1),
            )
            self.face = state.dealer

    def __str__(self):
        msg = [ '{0}: {1}'.format(*x) for x in zip(('p', 'd'), self.table) ]
        return ', '.join(msg)

    def play(self):
        episode = []

        for (i, p) in enumerate(self.table):
            while True:
                action = p.hit(self.face)
                if not i:
                    state = p.tostate(self.face)
                    episode.append((state, action))
                if not action:
                    break

                try:
                    p.deal(next(self.deck))
                except OverflowError:
                    reward = 1 if i else -1
                    return (episode, reward)

        naturals = list(map(bool, self.table))
        if all(naturals):
            reward = 0
        elif any(naturals):
            (p, _) = naturals
            reward = 1 if p else -1
        else:
            (p, d) = map(int, self.table)
            reward = (p > d) - (p < d) # https://stackoverflow.com/a/11215908

        return (episode, reward)

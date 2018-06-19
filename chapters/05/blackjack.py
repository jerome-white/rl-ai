import random
import collections as cl

Card = cl.namedtuple('Card', 'suit, value')
State = cl.namedtuple('State', 'player, dealer, ace')

class Deck(list):
    def __init__(self):
        for i in range(4):
            for j in range(13):
                value = min(j + 1, 10) if j else None
                self.append(Card(i, value))

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
        return self.cards < 2 or int(self) <= 21 and not self.stick()

    def deal(self, card):
        self.cards += 1

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
        return self.cards == 2 and int(self) == 21

    def stick(self):
        raise NotImplementedError()

class Dealer(Policy):
    def stick(self):
        return int(self) >= 17

class Player(Policy):
    def stick(self):
        return 20 <= int(self) <= 21

class Blackjack:
    def __init__(self, state=None):
        self.deck = Deck()

        if state is None:
            self.player = Player()
            self.dealer = Dealer()

            for (i, p) in enumerate((self.player, self.dealer)):
                for j in range(2):
                    card = next(deck)
                    p.deal(card)
                    if i and not j:
                        self.face = card
        else:
            self.player = Player(state.player, 2, state.ace)
            self.dealer = Dealer(state.dealer, 1)
            self.face = state.dealer

    def play(self):
        episode = []

        while True:
            state = State(int(self.player), self.face, self.player.ace)
            action = bool(self.player)
            episode.append((state, action))

            if not action:
                break

            try:
                self.player.deal(next(deck))
            except OverflowError:
                return (episode, -1)

        while self.dealer:
            try:
                self.dealer.deal(next(deck))
            except OverflowError:
                return (episode, 1)

        if self.player.isnatural():
            reward = int(not self.dealer.isnatural())
        else:
            (p, d) = [ int(x) for x in (self.player, self.dealer) ]
            reward = ((p > d) - (p < d)) # https://stackoverflow.com/a/11215908

        return (episode, reward)

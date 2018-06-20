import random
import collections as cl

Card = cl.namedtuple('Card', 'suit, value')
State = cl.namedtuple('State', 'player, dealer, ace')

class Deck(list):
    def __init__(self):
        for i in range(4):
            for j in range(1, 13):
                value = min(j, 10)
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

    def hit(self, facecard):
        return int(self) <= 21 and self._hit(facecard)

    def deal(self, card):
        self.cards += 1

        if card.value > 1:
            self.value += card.value
        else:
            if self.value + 11 <= 21:
                self.value += 11
                self.ace = True
            else:
                self.value += 1

        if self.value > 21:
            raise OverflowError()

    def isnatural(self):
        return self.cards == 2 and int(self) == 21

    def _hit(self, facecard):
        raise NotImplementedError()

class Dealer(Policy):
    def _hit(self, facecard):
        return int(self) < 17

class Player(Policy):
    def _hit(self, facecard):
        return int(self) < 20

class Blackjack:
    def __init__(self, state=None, player=Player):
        self.deck = Deck()

        if state is None:
            self.player = player()
            self.dealer = Dealer()

            for (i, p) in enumerate((self.player, self.dealer)):
                for j in range(2):
                    card = next(self.deck)
                    p.deal(card)
                    if i and not j:
                        self.face = card.value
        else:
            self.player = player(value=state.player, cards=2, ace=state.ace)
            self.dealer = Dealer(state.dealer, 1)
            self.face = state.dealer

    def __str__(self):
        iterable = map(int, (self.dealer, self.player))
        return 'd: {0:2d}, p: {1:2d}'.format(*iterable)

    def play(self):
        episode = []

        for (i, p) in enumerate((self.player, self.dealer)):
            while True:
                action = p.hit(self.face)
                if not i:
                    state = State(int(p), self.face, p.ace)
                    episode.append((state, action))
                if not action:
                    break

                try:
                    p.deal(next(self.deck))
                except OverflowError:
                    reward = 1 if i else -1
                    return (episode, reward)

        if self.player.isnatural():
            reward = int(not self.dealer.isnatural())
        else:
            (p, d) = map(int, (self.player, self.dealer))
            reward = (p > d) - (p < d) # https://stackoverflow.com/a/11215908

        return (episode, reward)

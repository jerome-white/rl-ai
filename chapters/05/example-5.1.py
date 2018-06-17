import itertools as it
import collections as cl

State = cl.namedtuple('State', 'player, dealer, ace')

def states():
    player = range(12, 21 + 1)
    dealer = range(1, 10 + 1)
    ace = (True, False)

    yield from it.starmap(State, it.product(player, dealer, ace))

def actions():
    yield from ('strike', 'stand')

for i in states():
    print(i)
    

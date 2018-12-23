import gridworld as gw

class Fall(gw.Wind):
    def __init__(self, shape):
        super().__init__()

        shape = [ getattr(shape, x) - 1 for x in ('row', 'column') ]
        (self.rows, self.columns) = shape

    def blow(self, state):
        if state.row == self.rows and 0 < state.column < self.columns:
            movement = -state.column
        else:
            movement = 0

        return (0, movement)

class Cliff(gw.GridWorld):
    def __init__(self, shape, start, goal):
        super().__init__(shape, goal, gw.FourPointCompass(), Fall(shape))
        self.start = start

    def navigate(self, state, action):
        (s, reward) = super().navigate(state, action)
        if state != self.start and s == self.start:
            reward *= 100

        return (s, reward)

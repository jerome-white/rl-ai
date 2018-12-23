import gridworld as gw

class Fall(gw.Wind):
    def __init__(self, shape):
        super().__init__()

        (self.rows, self.columns) = [ x - 1 for x in shape ]

    def blow(self, state):
        if state.row == self.rows and 0 < state.column < self.columns:
            movement = -state.column
        else:
            movement = 0

        return (0, movement)

class Cliff(gw.GridWorld):
    def __init__(self, shape, goal, start):
        super().__init__(shape, goal, gw.FourPointCompass(), Fall(shape))
        self.start = start

    def navigate(self, state, action):
        (state_, reward) = super().navigate(state, action)
        if state_ == self.start:
            reward *= 100

        return (state_, reward)

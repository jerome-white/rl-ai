import gridworld as gw

class Cliff(gw.GridWorld):
    def __init__(self, shape, goal, start):
        super().__init__(shape, goal, gw.FourPointCompass())

        self.start = start
        self.the_cliff = set()

        bottom = self.shape.row - 1
        for column in range(1, self.shape.column - 1):
            ledge = gw.State(bottom, column)
            self.the_cliff.add(ledge)

    def _navigate(self, state, action):
        if state in self.the_cliff:
            return (self.start, -100)
        else:
            reward = -int(state != self.goal)
            return (state, reward)

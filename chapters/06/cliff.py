import gridworld as gw

class Calm(gw.Wind):
    def blow(self, state):
        return state

class Cliff(gw.GridWorld):
    def __init__(self, rows, columns, goal, start):
        super().__init__(rows, columns, goal, gw.FourPointCompass(), Calm())
        self.start = start

        self.the_cliff = set()
        cliff = self.shape.rows - 1
        for i in range(1, self.shape.columns):
            state = gw.State(cliff, i)
            self.the_cliff.add(state)

    def navigate(self, state, action):
        state += action
        if state in self.the_cliff:
            state = self.start
            reward = -100
        else:
            reward = -1

        return (state, reward)

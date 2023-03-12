ACTION_SPACE = {'u', 'd', 'l', 'r'}


class Grid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state (self):
        return self.i, self.j

    def reset(self):
        self.i = 2
        self.j = 0
        return (self.i, self.j)

    def is_terminal(self, state):
        return state not in self.actions

    def get_next_state(self, state, action):
        i = state[0]
        j = state[1]
        if action in self.actions[(i, j)]:
            if action == 'u':
                i -= 1
            elif action == 'd':
                i += 1
            elif action == 'r':
                j += 1
            elif action == 'l':
                j -= 1
        return i, j

    def move(self, action):
        # Check if move is legal
        if action in self.actions[(self.i, self.j)]:
            if action == 'u':
                self.i -= 1
            elif action == 'd':
                self.i += 1
            elif action == 'r':
                self.j += 1
            elif action == 'l':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if action == 'u':
            self.i += 1
        elif action == 'd':
            self.i -= 1
        elif action == 'r':
            self.j -= 1
        elif action == 'l':
            self.j += 1

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
      (0, 0): ('d', 'r'),
      (0, 1): ('l', 'r'),
      (0, 2): ('l', 'd', 'r'),
      (1, 0): ('u', 'd'),
      (1, 2): ('u', 'd', 'r'),
      (2, 0): ('u', 'r'),
      (2, 1): ('l', 'r'),
      (2, 2): ('l', 'r', 'u'),
      (2, 3): ('l', 'u')
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    g = standard_grid()
    g.rewards.update({
      (0, 0): step_cost,
      (0, 1): step_cost,
      (0, 2): step_cost,
      (1, 0): step_cost,
      (1, 2): step_cost,
      (2, 0): step_cost,
      (2, 1): step_cost,
      (2, 2): step_cost,
      (2, 3): step_cost
    })
    return g

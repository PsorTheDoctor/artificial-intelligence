import numpy as np
from machine_learning.reinforcement_learning.gridworld.gridworld import (
  standard_grid, ACTION_SPACE
)
converge_thresh = 0.001


def print_values(V, grid):
    for i in range(grid.rows):
        print('------------------------')
        for j in range(grid.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(' %.2f|' % v, end='')
            else:
                print('%.2f|' % v, end='')
        print()


def print_policy(P, grid):
    for i in range(grid.rows):
        print('------------------------')
        for j in range(grid.cols):
            a = P.get((i, j), ' ')
            print('  %s  |' % a, end='')
        print()


if __name__ == '__main__':
    # Define transition probabilities and grid
    transition_probs = {}
    rewards = {}

    grid = standard_grid()
    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in grid.rewards:
                        rewards[(s, a, s2)] = grid.rewards[s2]

    policy = {
      (2, 0): 'u',
      (1, 0): 'u',
      (0, 0): 'r',
      (0, 1): 'r',
      (0, 2): 'r',
      (1, 2): 'u',
      (2, 1): 'r',
      (2, 2): 'u',
      (2, 3): 'l'
    }
    print_policy(policy, grid)

    V = {}  # Initialize V(s) = 0
    for s in grid.all_states():
        V[s] = 0

    gamma = 0.9  # Discount factor

    # Repeat until convergence
    iter = 1
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        print('Iter:', iter, 'biggest change:', biggest_change)
        print_values(V, grid)
        iter += 1

        if biggest_change < converge_thresh:
            break
    print()

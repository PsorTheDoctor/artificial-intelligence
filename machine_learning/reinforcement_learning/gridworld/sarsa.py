import numpy as np
import matplotlib.pyplot as plt
from machine_learning.reinforcement_learning.gridworld.gridworld import (
  negative_grid
)
from machine_learning.reinforcement_learning.gridworld.iterative_policy import (
  print_values, print_policy
)
from machine_learning.reinforcement_learning.gridworld.monte_carlo import (
  max_dict
)
gamma = 0.9
alpha = 0.1
action_space = ['u', 'd', 'l', 'r']


def epsilon_greedy(Q, s, eps=0.1):
    if np.random.random() < eps:
        return np.random.choice(action_space)
    else:
        a_optimal = max_dict(Q[s])[0]
        return a_optimal


if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)

    print('Rewards')
    print_values(grid.rewards, grid)

    # Initialize Q(s, a) = 0
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in action_space:
            Q[s][a] = 0

    update_counts = {}

    reward_per_episode = []
    for i in range(10000):
        if i % 2000 == 0:
            print('Iter:', i)

        # Begin a new episode
        s = grid.reset()
        a = epsilon_greedy(Q, s, eps=0.1)
        episode_reward = 0
        while not grid.game_over():
            # Perform action and get next state + reward
            r = grid.move(a)
            s2 = grid.current_state()
            # Update reward
            episode_reward += r
            # Get next action
            a2 = epsilon_greedy(Q, s2, eps=0.1)
            # Update Q(s, a)
            Q[s][a] += alpha * (r + gamma * Q[s2][a2] - Q [s][a])
            # Check how often Q(s) is updated
            update_counts[s] = update_counts.get(s, 0) + 1
            # Next state becomes current state
            s = s2
            a = a2

        # Log the reward for this episode
        reward_per_episode.append(episode_reward)

    plt.plot(reward_per_episode)
    plt.title('Reward per episode')
    plt.show()

    # Determine the policy from Q*
    # Find V* from Q*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    # The proportion of time we spend updating each part of Q
    print('Update counts:')
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total

    print_values(update_counts, grid)
    print('Values:')
    print_values(V, grid)
    print('Policy:')
    print_policy(policy, grid)

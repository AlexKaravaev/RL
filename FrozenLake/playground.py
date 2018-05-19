import gym
import numpy as np
import sys
from dynamic_programming.value_iter import value_iteration
from dynamic_programming.policy_iter import policy_iteration


verbose_str = str(sys.argv[1])
verbose = verbose_str == "v"
print("Output: {}".format(verbose))

action_map = {
    0: '\u2191', #up
    1: '\u2192', #right
    2: '\u2193', #down
    3: '\u2194', #left
}


def play_episodes(env, n_episodes, policy):
    wins = 0
    total_reward = 0

    for episode in range(n_episodes):

        terminated = False
        state = env.reset()

        # plat until get into terminal state
        while not terminated:

            if(verbose):
                env.render()

            #select best action
            action = np.argmax(policy[state])

            # and perfrom this action
            next_state, reward, terminated, info = env.step(action)

            total_reward += reward

            state = next_state

            if terminated and reward == 1.0:
                wins += 1

    avg_reward = total_reward / n_episodes

    return wins, total_reward, avg_reward

n_episodes = 10000

env = gym.make('FrozenLake8x8-v0')

# policy, V = value_iteration(env.env)
#
# print(''.join([action_map[action] for action in np.argmax(policy, axis=1)]))
#
# wins, total_reward, avg_reward = play_episodes(env, n_episodes, policy)
# print("{} wins out of {} episodes".format(wins, n_episodes))
# print("avg reward is {}".format(avg_reward))

policy, V = policy_iteration(env.env)

print(''.join([action_map[action] for action in np.argmax(policy, axis=1)]))

wins, total_reward, avg_reward = play_episodes(env, n_episodes, policy)
print("{} wins out of {} episodes".format(wins, n_episodes))
print("avg reward is {}".format(avg_reward))

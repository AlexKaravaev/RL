import numpy as np
import random
import gym

def q_learning(env, discount_factor = 0.95, max_iter = 2000, max_steps = 100, alpha = 0.8, eps = 1.0):
    """
    function that implements q-learning to solve mdp
    :param max_iter - maximum n_episodes
    :param max_steps - maximum steps per episode
    :param alpha - learning rate
    :param eps - exploration rate
    """

    # initialize Q-table
    Q = np.zeros([env.nS,env.nA])

    for i in range(int(max_iter)):
        state = env.reset()
        total_rewards = 0
        # do 100 steps
        for step in range(int(max_steps)):

            # check wether we going to do exploration or exploitation

            if random.random() > eps:
                # choose really best
                best_action = np.argmax(Q[state])
            else:
                # choose random
                best_action = random.randint(0,np.shape(Q)[1]-1)
            
            # calculate reward for best_action
            new_state, reward, done, _ = env.step(best_action)

            # update Q value
            Q[state][best_action] = Q[state][best_action] + alpha*(reward + discount_factor * np.max(Q[new_state]) - Q[state][best_action])

            # update state and total reward
            state = new_state
            total_rewards += reward

            if done:
                break

        # decay epsilon
        eps -= 1/max_iter

    return Q

def play(env, q_table, max_steps = 100, n_of_episodes = 500):
    wins = 0
    for i in range(n_of_episodes):
        state = env.reset()

        for step in range(max_steps):
            action = np.argmax(q_table[state])
            new_state, reward, done, _ = env.step(action)

            if done:
                wins += 1
                break
            state = new_state
    print("{}% of wins".format(wins/n_of_episodes * 100))


env = gym.make('FrozenLake-v0')
q_table = q_learning(env.env)
print(q_table)
play(env.env, q_table)

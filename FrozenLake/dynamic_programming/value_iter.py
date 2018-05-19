import numpy as np

def one_step_lookahead(env, state, V, discount_factor):
    action_values = np.zeros(env.nA)

    for action in range(env.nA):
        for prob, next_state, reward, _ in env.P[state][action]:
            action_values[action] += prob * (reward + discount_factor * V[next_state])

    return action_values


def value_iteration(env, discount_factor = 1.0, theta = 1e-9, max_iter = 1e9):
    """

    Value_iteration method to solve mdp.
    :param theta - stopping threshold.

    """

    # initialize array of values with zeroes for each state
    V = np.zeros(env.nS)

    for i in range(int(max_iter)):

        # difference between updated and old value
        delta = 0

        # update each state
        for state in range(env.nS):

            # update value of each value based on lookahead
            action_values = one_step_lookahead(env, state, V, discount_factor)

            #choose best
            best_action_value = np.max(action_values)

            # calculate difference
            delta = max(delta, np.abs(V[state] - best_action_value))

            #update value of the state
            V[state] = best_action_value

        if(delta < theta):
            print("Value iteration converged at {}".format(i))
            break


    # Let's create a policy given values of each state
    policy = np.zeros([env.nS, env.nA])

    for state in range(env.nS):

        # find the best action_values
        action_values = one_step_lookahead(env, state, V, discount_factor)

        # select the best action
        best_action = np.argmax(action_values)

        # update policy
        policy[state, best_action] = 1.0

    return policy, V

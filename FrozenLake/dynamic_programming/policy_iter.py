import numpy as np
from dynamic_programming.value_iter import one_step_lookahead

def policy_eval(policy, env, discount_factor = 1.0, theta = 1e-9, max_iter = 1e9):

    # number of iterations
    cur_iters = 1

    # init values
    V = np.zeros(env.nS)

    for i in range(int(max_iter)):

        # difference between updated and old value
        delta = 0

        # iterate for each state:
        for state in range(env.nS):

            # init a new val of cur_state
            v = 0

            # try all possible actions
            for action, action_prob in enumerate(policy[state]):

                # eval how good each next state will be
                for state_prob, next_state, reward, _ in env.P[state][action]:

                    # calculate expected value
                    v += action_prob * state_prob * (reward + discount_factor * V[next_state])


            # calculate delta
            delta = max(delta, np.abs(v - V[state]))

            # and update value func
            V[state] = v

        cur_iters += 1

        if delta < theta:
            print("Policy evaluated in {} iterations".format(cur_iters))
            return V


def policy_iteration(env, discount_factor = 1.0, max_iter = 1e-9):

    policy = np.ones([env.nS, env.nA]) / env.nA

    # count evalueated policies
    eval_policies = 1

    for i in range(int(max_iter)):

        stable = False

        # eval current
        V = policy_eval(policy, env, discount_factor = discount_factor)

        # go through each state and try to improve actions
        for state in range(env.nS):

            # choose best actions
            action = np.argmax(policy[state])

            action_value = one_step_lookahead(env, state, V, discount_factor)

            # select better action
            best_action = np.argmax(action_value)

            # if action did not change than policy is stable
            if best_action != action:
                stable = True

            # we are greedy
            policy[state] = np.eye(env.nA)[best_action]

        eval_policies += 1


        if stable:
            print('Evalueted {} policies'.format(eval_policies))
            return policy, V

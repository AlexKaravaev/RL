from nn import CNN
from env import create_env
from auxilary import stack_images, Memory
import vizdoom
import numpy as np
from collections import deque
import torch

##################################################
############### HYPERPARAMETERS ##################
##################################################

state_size = [84,84,4]                                      # image(state) size dimensions
action_size = game.get_available_buttons_size()             # available actions [left,right,shoot]
lr =  0.0002                                                # learning rate (alpha)

# Training params
total_episodes = 5000                                       # n of training episodes
max_steps = 100                                             # max possible steps in an episode
batch_size = 64                                             # batch_size for nn

# E-Greedy strategy params
explore_start = 1.0                                         # exploration probability at start
explore_stop = 0.01                                         # minimum exploration probability
decay_rate = 0.0001                                         # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99                                                # discount factor

# Memory params
pretrain_length = batch_size                                # length for pretraining filling of memory
memory_size = 50000                                         # memory size

### Preprocessing params
stack_size = 4                                              #
stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size), maxlen = 4])

##################################################
##################################################
##################################################

net = CNN()
mem = Memory(memory_size)
game, possible_actions = create_env()


# We need to pre-populate memory with taking some random actions
game.new_episode()

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state = stacked_frames(stacked_frames, state)

    # take random action
    action = random.choice(possible_actions)

    # observe reward from that action
    reward = game.make_action(action)

    # if episode is finished
    done = game.is_episode_finished()

    if done:
        next_state = np.zeros(state.shape)

        memory.add((state, action, reward, next_state, done))

        game.new_episode()

    else:
        next_state = game.get_state().screen_buffer
        next_state = stacked_frames(stacked_frames, next_state)

        memory.add((state, action, reward, next_state, done))

        state = next_state



#loss = torch.nn.L1Loss(size_average=True, reduce=True)

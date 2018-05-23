from nn import CNN
from env import create_env
from auxilary import stack_images
import vizdoom
import numpy as np

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
gamma = 0.99                                                #

# Memory params
pretrain_length = batch_size                                #
memory_size = 50000                                         #

### Preprocessing params
stack_size = 4                                              #

##################################################
##################################################
##################################################

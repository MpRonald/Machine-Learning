# -*- coding: utf-8 -*-

# -- Sheet --

# !pip install gym

import gym

env = gym.make('Taxi-v3').env
env.render()

env.reset()
env.render()

print('Possible actions: ', env.action_space)
print('Possible positions: ', env.observation_space)

# rendering scenery
state = env.encode(3,1,2,0) # taxi row, taxi col, passenger idx, destiny ix
print('State: ', state)
env.s = state
env.render()

# reward table
env.P[328]

# values means
v_table = env.P[328]
prob = v_table[0][0][0]
next_state = v_table[0][0][1]
reward = v_table[0][0][2]
finished = v_table[0][0][3]

print(f'Probability: {prob}, Next State: {next_state}, Reward: {reward} and Finished: {finished}')

env.s = 328 # defining env
frames = []
times = 0
punish, reward = 0, 0
finished = False

while not finished:
    action = env.action_space.sample() # random action
    state, reward, finished, info = env.step(action)

    if reward == -10:
        punish += 1

    # frames
    frames.append({
        'frame' : env.render(mode = 'ansi'),
        'state' : state,
        'action' : action,
        'reward' : reward
    })

    times += 1

print('Timesteps: ', times)
print('Punish: ', punish)

# interactive visualizations
from IPython.display import clear_output
from time import sleep

def print_frames(frames, seconds = 0.1):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"TimeSteps: {i+1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(seconds)


print_frames(frames, seconds=0.1)

# # Q-Table



import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table.shape

# # Exploring values
# After to explore random action, the q_values can , this format the agent will search for the best
# option of action determining the state.
# 
# There's a half term between explore and use, We want to prevent that the agent doesn't need to do it every time and maybe arrive 'overfitting'. we need to avoid it, we can use a parameter that calls 'epsilon' to balance the actions during agent train


# training agent

import random

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(10000):
    state = env.reset()
    times, punish, reward = 0,0,0
    finished = False

    while not finished:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() # explore
        else:
            action = np.argmax(q_table[state]) # use the best choice
        
        next_state, reward, finished, info = env.step(action)
        old_value = q_table[state, action]
        next_max_value = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max_value)
        q_table[state, action] = new_value

        if reward == -10:
            punish += 1
        
        state = next_state
        times += 1
    clear_output(wait=True)
    print('Episode: ', i+1)
print('Process finished!')

q_table




import gym 
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import pandas as pd
import time
# start timer
start = time.time()
# set seed to somewhat control the RNG
np.random.seed(5033)

# Environment set up
env = gym.make('CartPole-v1')

# Hyperparameters, can maybe do sensitivity analysis?
gamma = 0.9
alpha = 0.5
# exponential decay
decay = 0.001

# Training values
max_number_of_steps = 500  # maximum length of each episode
num_consecutive_iterations = 100  # number of episodes used to get average reward
num_episodes = 10000  # Total number of episodes
max_reward = 475  # maximum rewards value

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
num_bins = 6
num_states = num_bins ** state_size
q_table = np.zeros((num_states, action_size)) # initialize q table with all 0s

total_reward_vec = np.zeros(num_consecutive_iterations)  # store the average reward
episodelist = []
scorelist = []
steplist = []

# Define the number of bins/categories 
# for discretization
def bins(min, max, num):
    return np.linspace(min, max, num + 1)[1:-1]

# Discretizing the state
# Because CartPole has continuous state
# We need to discretize it
def discretized(observation):
    # print(observation)
    cart_pos, cart_v, pole_angle, pole_v = observation
    # make the observation space from continuous to discrete
    # for cart_v and pole_v I'm playing around with the bounds
    discretized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_bins)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_bins)),
        np.digitize(pole_angle, bins=bins(-0.2095, 0.2095, num_bins)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_bins))
    ]
    return sum([x * (num_bins**i) for i, x in enumerate(discretized)])

# Epsilon-greedy method
def epsilon_greedy(next_state):
    # static
    epsilon = 0.5
    # exponential decay
    #epsilon = 0.5 * math.exp(-decay*episode)
    # episode decay
    #epsilon = 0.5 * (1/episode)
    if epsilon <= np.random.uniform(0, 1): #exploitation
        next_action = np.argmax(q_table[next_state])
    else: # exploration
        next_action = env.action_space.sample()
    
    return next_action

# softmax method
def soft_max(next_state):
    # softmax  -----
    a_choice = 0,1
    t = 0.1
    exp_num = np.array([np.exp(q_table[next_state][a]/t) for a in a_choice])
    total_epx_denom = np.sum([np.exp(q_table[next_state][b]/t) for b in a_choice])
    probability = np.cumsum(exp_num / total_epx_denom)
    next_action = np.argwhere(probability > np.random.uniform(0,1))[0][0]
    return next_action

# Q-table update, uncomment the correct block for whichever algorithm we need
def update_q(q_table, state, action, reward, next_state, next_action):
    # Our friend Bellman
    # Q function
    # from the book and slide

    # for SARASA
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state,action])

    #for Q-Learning
    #q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max(q_table[next_state][0],q_table[next_state][1]) - q_table[state, action])
    return q_table

# Main loop
for episode in range(1, num_episodes+1):  # repeat for the number of trials
    # Initialize the environment
    observation = env.reset()
    # Initialize S, discretized state space
    state = discretized(observation[0])
    # Choose A from S using epsilon greedy or softmax
    # Uncomment the desired policy
    action = epsilon_greedy(state)
    #action = soft_max(state)
    # Initial reward
    episode_reward = 0

    # loop for trials, we only care if the steps go until the maximum reward
    for t in range(max_reward + 1):
        # Take an action, and observe reward, next step, etc.
        # for some reason, not having 'extra' throws an error
        observation, reward, done, info, extra = env.step(action)

        # Set a penalty
        # Otherwise, epsilon greedy will not be efficient
        if done and t < max_reward:
            reward = -10  # penalty if the episode fails, otherwise, reward will be +1 

        episode_reward += reward  # add reward
        # Get the next state
        next_state = discretized(observation)  # convert the observation state at t+1 to a discrete value

        # Get the next action from the next state, using epsilon greedy or softmax
        # Again, simply uncomment the desired policy
        next_action = epsilon_greedy(next_state) # get the next action
        #next_action = soft_max(next_state)
        
        # Update our Q values
        q_table = update_q(q_table, state, action, reward, next_state, next_action)

        # Update the next action and state
        action = next_action
        state = next_state
        
        # Print out our results
        if done:
            episodelist.append(episode)
            scorelist.append(total_reward_vec.mean())
            steplist.append(t)
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # record reward
            print('Episode %d finished after %d time steps / with score %d and mean %f' %
                  (episode, t, episode_reward, total_reward_vec.mean()))
            break

# end timer
end = time.time()

# calculate total time to run 10000 episodes
total_time = end - start
print('Total time to run 10000 episodes is', str(total_time))

mean = sum(scorelist)/len(scorelist)
stdev = statistics.pstdev(scorelist)

print('Mean:', mean)
print('Standard deviation:', stdev)

# these lines are for generating csv files for comparison plots
#meandf = pd.DataFrame(scorelist)
#meandf.to_csv('static-q-learning.csv')

# Plotting

# Average score per episode
plt.plot(episodelist, scorelist)

# Axis labelling 
plt.xlabel('episode')
plt.ylabel('average reward')

# Plot title and showing the plot
plt.title('Average Reward per Episode')
plt.show()
import gym 
from gym import wrappers # for image savings
import numpy as np
import time
import matplotlib.pyplot as plt

# set seed to somewhat control the RNG
np.random.seed(5033)
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
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_discretized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_discretized)),
        np.digitize(pole_angle, bins=bins(-0.2095, 0.2095, num_discretized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_discretized))
    ]
    return sum([x * (num_discretized**i) for i, x in enumerate(discretized)])

# Epsilon-greedy method
def get_action(next_state, episode):  # Gradually take only optimal actions, epsilon-greedy method
    epsilon = 0.5 * (1 / (episode + 1)) # define some epsilon value, can maybe do sensitivity analysis?
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action

# Q-table
def update_q(q_table, state, action, reward, next_state, next_action):
    # Our friend Bellman
    # Q function
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])

    return q_table

# Environment set up
env = gym.make('CartPole-v1')
gamma = 0.9
alpha = 0.5
max_number_of_steps = 500  # maximum length of each episode
num_consecutive_iterations = 100  # Number of trials used to evaluate learning completion
num_episodes = 850  # Total number of trials
max_reward = 475  # maximum rewards value
num_discretized = 6  # Number of divisions of the state
q_table = np.random.uniform(low=-1, high=1, 
    size=(num_discretized**4, env.action_space.n))
total_reward_vec = np.zeros(num_consecutive_iterations)  # Store the reward of each trial

episodelist = []
scorelist = []
steplist = []
# Main loop
for episode in range(1, num_episodes+1):  # repeat for the number of trials
    # Initialize the environment
    observation = env.reset()
    # Discretize the observation space
    state = discretized(observation[0])
    # Choose an action
    action = np.argmax(q_table[state])
    # Initial reward
    episode_reward = 0
    #print(episodelist)

    # loop for trials
    for t in range(max_number_of_steps):
        # get the next state, reward, etc.
        # for some reason, not having 'extra' breaks the code
        observation, reward, done, info, extra = env.step(action)

        # Set reward and penalty
        if done:
            if t < 475:
                reward = -10  # penalty if the episode fails
            else:
                reward = 1  # no penalty if it remains upright
        else:
            reward = 1  # reward for standing at each step

        episode_reward += reward  # add reward
        # Calculate discrete state s_{t+1}
        next_state = discretized(observation)  # convert the observation state at t+1 to a discrete value

        next_action = get_action(next_state, episode) # get the next action
        q_table = update_q(q_table, state, action, reward, next_state, next_action) # update the q values

        # Update the next action and state
        action = next_action
        state = next_state

        # Processing at the end
        if done:
            print('Episode %d finished after %d time steps / with score %d and mean %f' %
                  (episode, t, episode_reward, total_reward_vec.mean()))
            episodelist.append(episode)
            scorelist.append(total_reward_vec.mean())
            steplist.append(t)
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # record reward
            break


plt.plot(episodelist, steplist)
plt.plot(episodelist, scorelist)
 
plt.xlabel('episode')
plt.ylabel('number of steps/average reward')

plt.legend(['number of steps', 'average reward'], loc = 'upper left')

plt.title('Number of Steps and Average Reward per Episode')
plt.show()
# if islearned:
#    np.savetxt('final_x.csv', final_x, delimiter=",")  # save the final x-coordinate
import gym 
from gym import wrappers # for image savings
import numpy as np
import time

# Define the number of bins/categories 
# for discretization
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# Discretizing the state
# Because CartPole has continuous state
# We need to discretize it
def digitize_state(observation):
    # print(observation)
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])

# Epsilon-greedy method
def get_action(next_state, episode):  # Gradually take only optimal actions, epsilon-greedy method
    epsilon = 0.5 * (1 / (episode + 1)) # define some epsilon value
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action

# Q-table
def update_q(q_table, state, action, reward, next_state, next_action):
    gamma = 0.99
    alpha = 0.5
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])

    return q_table

# Set up parameters --------------------------------------------------------
env = gym.make('CartPole-v1')
max_number_of_steps = 500  # maximum length of each episode
num_consecutive_iterations = 100  # Number of trials used to evaluate learning completion
num_episodes = 1000  # Total number of trials
goal_average_reward = 475  # maximum rewards value
num_dizitized = 6  # Number of divisions of the state
q_table = np.random.uniform(low=-1, high=1, size=(num_dizitized**4, env.action_space.n))
total_reward_vec = np.zeros(num_consecutive_iterations)  # Store the reward of each trial
final_x = np.zeros((num_episodes, 1))  # Store the x position of the cart at t=200 after learning
islearned = 0  # Flag to check if learning is done
isrender = 0  # Rendering flag

# [5] Main routine--------------------------------------------------
for episode in range(num_episodes):  # repeat for the number of trials
    # Initialize the environment
    observation = env.reset()
    state = digitize_state(observation[0])
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):  # loop for one trial
        if islearned == 1:  # if learning is finished, render cartPole
            env.render()
            time.sleep(0.1)
            print(observation[0])  # output the x position of the cart

        # Calculate s_{t+1}, r_{t}, etc. by executing action a_t
        observation, reward, done, info, extra = env.step(action)

        # Set and give reward
        if done:
            if t < 475:
                reward = -10  # penalty if it falls down
            else:
                reward = 1  # no penalty if it remains upright
        else:
            reward = 1  # reward for standing at each step

        episode_reward += reward  # add reward

        # Calculate discrete state s_{t+1}
        next_state = digitize_state(observation)  # convert the observation state at t+1 to a discrete value

        # *This is different from Q-learning*
        next_action = get_action(next_state, episode)  # calculate the next action a_{t+1}
        q_table = update_q(q_table, state, action, reward, next_state, next_action)

        # Update the next action and state
        action = next_action  # a_{t+1}
        state = next_state  # s_{t+1}

        # Processing at the end
        if done:
            print('Episode %d finished after %d time steps / with score %d and mean %f' %
                  (episode, t + 1, episode_reward, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # record reward
            if islearned == 1:  # if learning is finished, store the final x-coordinate
                final_x[episode, 0] = observation[0]
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # if the recent 100 episodes have achieved the goal reward or more, the training is successful
        print('Episode %d train agent successfully!' % episode)
        islearned = 1
        #np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #if you want to save the Q table
        if isrender == 0:
            #env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #if you want to save a video of the result
            isrender = 1

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")  # save the final x-coordinate
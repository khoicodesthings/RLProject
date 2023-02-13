import gym
import numpy as np

# Define the environment
env = gym.make('CartPole-v1')

# Define the learning rate and discount factor
alpha = 0.1
gamma = 0.99

# Define the weight matrix
weights = np.zeros((2, 4))

# Define the number of episodes to run
num_episodes = 500

# Define the TD learning loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        # Choose an action using a softmax policy
        state = np.expand_dims(state, axis=0)
        state = state.flatten()
        action_probs = np.exp(weights.dot(state)) / np.sum(np.exp(weights.dot(state)), axis=1)
        action_probs = action_probs.flatten()
        action = np.random.choice(2, p=action_probs)
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        # Update the weights
        next_state = np.expand_dims(next_state, axis=0)
        td_error = reward + gamma * np.max(weights.dot(next_state.T), axis=1) - weights[action].dot(state.T)
        weights[action] += alpha * td_error.flatten() * state
        state = next_state
    print("Episode {} finished with reward {}".format(episode, episode_reward))

# Close the environment
env.close()

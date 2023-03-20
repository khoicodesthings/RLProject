# import libraries
import gym  
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import csv

# setting -----------------------------------------------------------------
# env = gym.make('CartPole-v1', render_mode="human")
env = gym.make('CartPole-v1' )
max_number_of_steps = 500         # max steps of each episode
num_consecutive_iterations = 100  # Number of trials used to evaluate 
num_episodes = 10000              # Total number of trials
goal_avgR = 475                   # maximum rewards value
num_dizitized = 6                 # Number of divisions of the state
q_table = np.random.uniform(low=-1, high=1, size=(num_dizitized**4, env.action_space.n))

total_reward_vec = np.zeros(num_consecutive_iterations)  # store the reward of each trial
final_x = np.zeros((num_episodes, 1))

islearned = 0  #flag when learning is done 
isrender = 0   #flag for render 

np.random.seed(5033)



# create bins　----------------------------------------------------------
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# Discretizing the state ------------------------------------------------
def discretize_s(obs):
    cart_pos, cart_v, pole_angle, pole_v = obs
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.2095, 0.2095, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])

# get action-value function using epsilon-greedy method ------------------
def get_greedy_act(next_state,episodes):
    # use epsilon-greedy method to get optimal actions
    
    ## randomized algorithm -----
    # a = 0,1
    # pi_num = np.cumsum(np.tile(1, len(a)) / len(a))
    # return np.argwhere(pi_num > np.random.uniform(0,1))[0][0]
    
    ## epsilon decay1 -----
    #epsilon = 0.5*math.exp(-0.001 * episodes)
    ## epsilon decay2 -----
    #  epsilon = 0.5 * (1 / (episode + 1)) 
    ## epsilon decay3 -----
    # epsilon = 0.3
    # eps_decay = 0.99
    # epsilon = max(epsilon * eps_decay, 0.01)
    
    ## select best action epsilon greedy with decay -----
    # greedy = np.random.random()
    # eps = 0.01
    # if greedy < eps:
    #     next_action = np.random.randint(2)
    # else:
    #     next_action = np.argmax(q_table[next_state])
    #return next_action

    
    #----------------------------------------------------------------
    
    # exponential decay
    # decay = 0.001
    # epsilon = 0.5 * math.exp(-decay*episode)
    
    # episode decay
    epsilon = 0.5 * (1/(episodes+1))
    
    ## softmax  -----
#     a_choice = 0,1
#     t = 0.1
#     exp_num = np.array([np.exp(q_table[next_state][a]/t) for a in a_choice])
#     total_epx_denom = np.sum([np.exp(q_table[next_state][b]/t) for b in a_choice])
#     pi_val = np.cumsum(exp_num / total_epx_denom)
#     next_action = np.argwhere(pi_val > np.random.uniform(0,1))[0][0]
#     return next_action
    

    ## epsilon greedy -----
    #epsilon = 0.5
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action
    
    

# update q table using Q-learning -----------------------------------------
def update_q(q_table, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.5
    next_maxQ = max(q_table[next_state][0],q_table[next_state][1] )
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * next_maxQ - q_table[state, action])
   
    return q_table

# Main loop ---------------------------------------------------------------

num_loop = 10
for n in range(num_loop):
    
    store_r = []
    store_epi = []
    store_steps = []
    total_r = []
    
    for episode in range(num_episodes):  # repeat for the num of episodes 
        # Initialize the encironment 
        observation = env.reset()
        # Initialize state 
        state = discretize_s(observation[0])

        action = np.argmax(q_table[state])
        # Initial reward 
        episode_reward = 0

        # loop for trials
        for t in range(max_number_of_steps+1):
            if islearned == 1:  # draw cartPole when training is done 
                env.render()
                time.sleep(0.1)
                # print (observation[0])  # print cart x psition  

            observation, reward, done, info, extra = env.step(action)

            # Set reward and penalty
            if done:
                if t < 475:
                    reward = -10  # penalty if the episode fails
                else:
                    reward = 1  # no penalty if it remains upright
            else:
                reward = 1  # add a reward for standing at each step

            episode_reward += reward  # add reward 

            # get new discrete state and update q table 
            next_state = discretize_s(observation)  # convert 
            q_table = update_q(q_table, state, action, reward, next_state)

            # get next action a_{t+1}
            action = get_greedy_act(next_state, episode)    # a_{t+1} 
            state = next_state
            

            # process for finishing  
            if done:
                #print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                store_r.append(total_reward_vec.mean())
                store_epi.append(episode)
                store_steps.append(t)
                total_reward_vec = np.hstack((total_reward_vec[1:],episode_reward))  #record reward 

                if islearned == 1:  # when finished training, store final x position 
                    final_x[episode, 0] = observation[0]
                break
                
        

        if (total_reward_vec.mean() >= goal_avgR):  # when training is succeed
            print('Episode %d train agent successfuly!' % episode)
            islearned = 1
            np.savetxt('learned_Q_table.csv',q_table, delimiter=",") # store Qtable
            if isrender == 0:
                isrender = 1
                
    #filename = 'eps_greedy_{0}.csv'.format(n)
    #filename = 'expo_decay{0}.csv'.format(n)
    filename = 'episode_decay{0}.csv'.format(n)
    #filename = 'softmax_new{0}.csv'.format(n)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(map(lambda r: [r], store_r))
    print("finished{0} test".format(n))
    
    plt.plot(store_epi, store_r)
    plt.title("Q-Learning")
    plt.xlabel("episode")
    plt.ylabel("avgReward ")
    plt.legend(['average reward'], loc = 'upper left')
    #pic_name = 'eps_perf{0}.jpg'.format(n) 
    #pic_name = 'expo_decay_perf{0}.jpg'.format(n) 
    pic_name = 'episode_decay_perf{0}.jpg'.format(n) 
    #pic_name = 'softmax_perf_new{0}.jpg'.format(n) 
    plt.savefig(pic_name)
    
    


# if islearned:
#     np.savetxt('final_x.csv', final_x, delimiter=",")

# plot results 
#plt.xlabel("episode")
#plt.ylabel("num_steps")
#plt.show()
#ax = plt.subplot(111)

#plt.plot(store_epi, store_steps)
# plt.plot(store_epi, store_r)
# plt.title("Q-Learning")
# plt.xlabel("episode")
# plt.ylabel("avgReward ")
# plt.legend(['average reward'], loc = 'upper left')
# plt.savefig("performance_ql.jpg")




plt.show()

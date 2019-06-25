from __future__ import print_function
import gym
import matplotlib.pyplot as plt
from Ai_brain import deep_qlearn


#Hyperparameters
#####################################
input_space = 4
output_space = 2
gamma_discount_value =0.9
no_of_episodes = 500
timesteps_per_episode = 1000


#Defining the model
#####################################
brain = deep_qlearn(input_space, output_space, gamma_discount_value)
#####################################


#Initializing the values
#####################################
brain.aux_logits = False
scores = []
reward_window = []
#####################################

#Setting up the cartpole environment
#####################################
env = gym.make('CartPole-v0')
env.reset()
#####################################

#Loading already trained model
#####################################
brain.load('E:\cartpole\cartpole_model.pth')
#####################################

#Main-Code
#####################################
for i_episode in range(no_of_episodes):  #For loop to define maximum number of episodes that can be played by the agent.
    observation = env.reset()
    reward = 1
    done = False
    scores.append(sum(reward_window)) #for calculating maximum rewards a single episode gets, our agent's aim is to maximize this value
    reward_window = []
    for t in range(timesteps_per_episode): #For loop to define maximum number of timesteps a single episode that can be played
        env.render()
        if done:
            break
        action = brain.update(reward, observation)
        observation, reward, done, info = env.step(action.item())
        reward_window.append(reward)
#####################################

#Save the model
#####################################
brain.save('E:\cartpole\cartpole_model.pth')
#####################################

#closing the environment
#####################################
env.close()
#####################################

#Plotting the scores to visualize how the agent have performed.
#####################################
plt.plot(scores)
plt.show()
#####################################
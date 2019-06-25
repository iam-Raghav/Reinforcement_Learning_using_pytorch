from __future__ import print_function
import random
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Declaring and initializing the Module class
#####################################
class Net(nn.Module):
    def __init__(self,inp_obs_size, out_act_space):
        super(Net,self).__init__()
        self.inp_obs_size = inp_obs_size
        self.out_act_space = out_act_space
        self.fc1 = nn.Linear(inp_obs_size,40)
        self.fc2 = nn.Linear(40,out_act_space)

    def forward(self, obs_space):

        x = F.relu(self.fc1(obs_space)) #connecting the input layer to the hidden layer
        q_vals = self.fc2(x.view(x.size(0), -1)) #connecting the hidden layer to the output layer

        return q_vals
#####################################


#Experience replay class will help the agent to learn from its own action
#####################################
class exp_replay_mem(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
#Push function is called whenever a new memory state is inserted, its like inserting a row in a table
#####################################
    def push(self, curr_obs):
        self.memory.append(curr_obs)

        if len(self.memory) > self.capacity:
            del self.memory[0]

#####################################
#Random sample is similar to a pop function but it pops a random number of samples.
#####################################
    def random_sample(self, sample_size):

        samples = zip(*random.sample(self.memory, sample_size))
        return map(lambda x: t.cat(x), samples)
#####################################


class deep_qlearn():
    def __init__(self, inp_obs_size, out_act_space, gamma):
        self.inp_obs_size = inp_obs_size
        self.gamma = gamma
        self.out_act_space = out_act_space
        self.memory = exp_replay_mem(10000) #maximum memory states can be assigned here
        self.model = Net(inp_obs_size,out_act_space)
        self.optimizer = optim.Adam(self.model.parameters()) # Adam optimizer is used here but other optimizers can also be tried for better result
        #initializing the states, action and rewards
        #####################################
        self.last_state = t.zeros([1,inp_obs_size], dtype = t.float, requires_grad= True)
        self.last_action = t.zeros([1,1], dtype = t.int)
        self.last_reward = t.zeros([1,1], dtype = t.float)
        #####################################

#select_action-This function is used both during training and live selection of actions during game episodes.
#####################################
    def select_action(self, state):

        action =nn.Softmax(dim =1)
        out_action  = action(self.model(t.tensor(state)))
        return out_action.max(1)[1].view(1,1)

#####################################
#Learn- This is where pretty much the learning happens(calculating loss, backpropagation and updating the weights)
#####################################
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):

        outputs = self.model(batch_state).gather(1,batch_action)
        next_state_outputs= self.model(batch_next_state).max(1)[0].detach()
        target = batch_reward + (self.gamma*next_state_outputs) #Bellman equations
        loss = F.smooth_l1_loss(outputs, target.unsqueeze(1)) #smooth_l1_loss is used here, other loss function can be experimented
        self.optimizer.zero_grad()
        loss.backward() #this is where backpropagation happens
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() #this is step where the updation of weights happen

#####################################
#Update-This is function which push the last state, new state , last reward and last action into the memory
#####################################
    def update(self, reward, new_obs_space):

        new_state = t.tensor(new_obs_space, dtype = t.float, requires_grad= True).unsqueeze(0)
        reward = t.tensor([reward]).unsqueeze(0)
        self.memory.push([self.last_state, new_state, t.tensor(self.last_reward).float(), t.tensor((self.last_action), dtype= t.long)])
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.random_sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        return action

#####################################

#Save-this function saves the model
####################################
    def save(self,path):
        t.save(self.model.state_dict(), path)
        print('model saved')

#####################################

#Load- This function loads the existing model with already save model parameters.
#####################################
    def load(self, path):
        self.model.load_state_dict(t.load(path))
        self.model.eval()
        print('model load')
#####################################















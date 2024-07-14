import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from DQN import DQN
from DQN import ReplayBuffer
#DDQN
#Double DQN decouples action selection from action evaluation in training
#So we mofify the train function
#DDQN CHANGES IN TRAINING FUNCTION
class DDQNAgent:
    def __init__(self, state_shape, num_actions, buffer_size, batch_size, gamma, alpha, epsilon, epsilon_decay, min_epsilon):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.use_target_model = True
        self.min_epsilon = min_epsilon
        self.model = DQN(state_shape, num_actions) 
        self.target_model = DQN(state_shape, num_actions) 
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025) 
        self.loss_fn = nn.MSELoss() 

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) 

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0) 
        with torch.no_grad():
            q_values = self.model(state) 
        return torch.argmax(q_values).item() 

    def remember(self, state, action, reward, next_state, done): 
        self.memory.add((state, action, reward, next_state, done)) 

    def train(self):

        batch = self.memory.sample(self.batch_size) #sample random batch of experiences
        states, actions, rewards, next_states, dones = zip(*batch) #get all states, actions,... in order

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states) #Compute Q values for the sampled states with NN
        q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1) #retrieve Q values for taken actions

        # Double DQN Implementation
        #Decoupling of action selection and action evaluation: the main network now will estimate the best index and feed it to the target model
        #so we don't use MAX value anymore, but we feed ARGMAX to the target model, this should lower overestimation
        #Contrast this with the previous implementations of DQN: 
          #max_next_q_values = self.target_model(next_states).max(1)[0] when target model is used
          #max_next_q_values = self.model(next_states).max(1)[0] for basic DQN
        #We now stop using max(*)[0] and use argmax (max(*)[1])
        next_action_indices = self.model(next_states).max(1)[1] # Get the indices of the actions with max Q-value from the primary model, max(*)[1] gives index of max value
        max_next_q_values = self.target_model(next_states).gather(1, next_action_indices.unsqueeze(1)).squeeze(1) # Use target model to get Q-values of next states for these actions
        target_q_values = (1 - self.alpha ) * q_values + self.alpha * (rewards + self.gamma * max_next_q_values * (1 - dones)) # Q value equation to compute target q values

        loss = self.loss_fn(q_values, target_q_values.detach()) #compute loss between actual Q value and calculated one

        self.optimizer.zero_grad() #employ adam optimizer
        loss.backward()  #backward propagation to compute gradients and update weights
        self.optimizer.step()
        return loss.item() #return loss value to plot loss

    def save(self, filename):
      checkpoint = {
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'epsilon': self.epsilon,
          'gamma': self.gamma,
          'epsilon_decay': self.epsilon_decay,
          'min_epsilon': self.min_epsilon,
          'batch_size': self.batch_size
      }
      torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.gamma = checkpoint['gamma']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.min_epsilon = checkpoint['min_epsilon']
        self.batch_size = checkpoint['batch_size']

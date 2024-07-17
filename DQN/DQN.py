import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#A NN with two layers to start with
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions): #input shape is 10x2 and num_actions is 2
        super(DQN, self).__init__()
        self.flatten = nn.Flatten() #flatten observation into a 20 element vector, state is input
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, num_actions) #output each time is the actions with the estimated rewards

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) #use ReLu for non linear activation function
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) #output Q value we use linear unit in output layer
        return x

# Experience Replay Buffer to store and sample experiences, in place of the Q-table, use it for training purposes!
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size) #buffer is a queue data structure with max len

    def add(self, experience):
        self.buffer.append(experience) #add experience

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) #sample a random batch of experiences, batch size can be 16, 32, 64, ...

    def size(self):
        return len(self.buffer) #get current buffer size

# DQN Agent, this class handles interaction with the environment according to what action the NN outputs, wrapper of above classes
class DQNAgent:
    def __init__(self, state_shape, num_actions, buffer_size, batch_size, gamma, alpha, epsilon, epsilon_decay, min_epsilon, use_target_model=False):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.use_target_model = use_target_model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.model = DQN(state_shape, num_actions) #above model definition, input is state and output is actions with estimated rewards
        self.target_model = DQN(state_shape, num_actions) #copy weights from Q network to target network
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025) #optimizer
        self.loss_fn = nn.MSELoss() #use mean squared error as loss with gradient descent or Hubert Loss function

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) #hard update, copy weights from primary to target network

    #Greedy epsilon exploration or NN
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0) #convert state to a tensor
        with torch.no_grad():
            q_values = self.model(state) #compute actions with the NN and their Q value
        return torch.argmax(q_values).item() #return highest value action one (one integer) 

    def remember(self, state, action, reward, next_state, done): #memory replay allows us to remember <state, action, reward, next_state, termination>
        self.memory.add((state, action, reward, next_state, done)) #store experience in replay buffer

    #sample batch from replay buffer and then train NN, so the NN will compress the table representation.
    def train(self):

        batch = self.memory.sample(self.batch_size) #sample random batch of experiences
        states, actions, rewards, next_states, dones = zip(*batch) 

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states) #Compute Q values for the sampled states with NN
        q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1) #retrieve Q values for taken actions

        #If we use the target module, that will calculate the next q values, else we stick with our NN
        if self.use_target_model:
          max_next_q_values = self.target_model(next_states).max(1)[0] #Compute Q values for next states and get the max q value
          target_q_values = q_values * (1 - self.alpha) + self.alpha * (rewards + self.gamma * max_next_q_values * (1 - dones)) #Q value equation to compute target q values
        else:
          max_next_q_values = self.model(next_states).max(1)[0] #get action with max Q value estimated without target model
          target_q_values = q_values * (1 - self.alpha) + self.alpha * (rewards + self.gamma * max_next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach()) #compute loss between actual Q value and calculated one

        self.optimizer.zero_grad() #employ adam optimizer
        loss.backward()  #backw-propagation to compute gradients and update weights
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

# Example training loop, when training we estimate the Q function through exploration, so the network  will estimate best action to take
def train_dqn(env, agent, episodes): 
    rewards = []
    actions = []
    total_reward = 0
    avg_losses = []
    for episode in range(episodes):
        losses = []
        state, info = env.reset()
        state = np.reshape(state, [1, *state.shape]) 
        done = False
        while not done:
            action = agent.act(state)
            actions.append(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, *next_state.shape])
            agent.remember(state, action, reward, next_state, done) #Save the experience in the replay buffer
            state = next_state
            total_reward += reward
            if agent.memory.size() > agent.batch_size:
              loss = agent.train() #train NN for computing actions at every step
              losses.append(loss)

        avg_losses.append(np.mean(losses))
        if agent.use_target_model:
            agent.update_target_model() #update target model at every episode
        print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon} MemSize: {agent.memory.size()}")
        rewards.append(total_reward)
        total_reward = 0
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= agent.epsilon_decay

    return (rewards, avg_losses, actions)

def test_dqn(env, agent, episodes, render):
    agent.epsilon = 0.0  # Disable exploration
    total_reward = 0
    for episode in range(episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, *state.shape])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, *next_state.shape])
            state = next_state
            total_reward += reward
            print("Chosen action is ", action)
            frame = env.render()
            render.append(frame)
            if done:
                print(f"Episode: {episode + 1}/{episodes}, Reward: {total_reward}")
                break
    return total_reward / episodes


def get_reward_evolution(rewards): #Calculates average reward for 1 episode over the last 100 episodes
  avgs = []
  for r in range(len(rewards)):
    if(r < 99):
      continue
    else:
      res = 0
      for i in range(r-99, r):
        res += rewards[i]
      res = res / 100
      avgs.append(res)
  return avgs

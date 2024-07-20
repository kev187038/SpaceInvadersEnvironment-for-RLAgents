import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import gymnasium as gym

class EpsGreedyPolicy:
  def __init__(self, show):
    self.show = False
  def __call__(self, obs, eps, Q, env) -> int:
    """
    Q is our Q-Table
    Bernoulli distribution on greedy probability, epsilon is probability of a random action, initially 1
    For the observation we just did, seek best chances you can if you are greedy(action with max reward)
    If we are not greedy we explore doing a random action.
    """
    greedy = random.random() > eps
    if greedy:
      if np.all(Q[obs] == Q[obs][0]): #if the array is all zeroes do not pick first action but a random one to help with learning on greedy policy
        return np.random.choice(len(Q[obs]))  #pick random element from 0 to 5
      else:
        if self.show:
          print("Chosen action ", np.argmax(Q[obs]))
        return np.argmax(Q[obs])
    else:
      return env.action_space.sample()

class QTable:

  def __init__(self, eps=1.0, showAction=False, env=None):
    #Create Q-Table and assign policy
    #The Q-Table is designed as a dictionary of [observation][values]:reward, with the value being a numpy array, we can assign Q-Table without knowing observation space
    #load with zeroes if cell doesn't exist, fill in even if I don't know the keys
    #Each state key is a 128 values tuple, each action a number associated with a reward
    self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
    self.eps = eps
    self.policy = EpsGreedyPolicy(show=showAction)

  def qlearn(
      self,
      env,
      alpha0,
      gamma,
      max_steps
  ):
    """
    Q-Learning Algorithm
    Args:
      env: environment
      alpha0: learning rate
      gamma: discounting for further rewards
      n_episodes:
      max_steps:

    Returns:
      Q Table

    """
    rewards = []
    obs, info = env.reset()
    obs = bytes(obs)
    done = False
    tot_rew = 0
    for step in tqdm(range(max_steps)):
    #Do not stop when learning if environment finishes, we need to do it for n episodes
      if done:
        rewards.append(tot_rew) #reward gained for the episode 
        tot_rew = 0
        obs, info = env.reset()
        obs = bytes(obs) #we use a bytes representation of the RAM (dict key), since a tuple representation is wasteful (an observation array returned from environment is 240 bytes, this representation consumes 161 bytes per obs instead of 1064 of the tuple)
        done = False
      self.eps = (max_steps - step) / max_steps #Redefine eps each time (linear descent to ~0)
      action = self.policy(obs, self.eps, self.Q, env) #Call policy to get action
      #Do action
      results = env.step(action)

      #Get results for different environments
      if len(results) == 5:
        obs2, reward, terminated, truncated, info = results
      else:
        obs2, reward, terminated, info = results
        truncated = False

      obs2 = bytes(obs2)
      done = terminated or truncated #if the agent loses or wins the game is over
      max_next_q = np.max(self.Q[obs2])
      tot_rew += reward

      #Update the Q-Table and current observation to next observation
      self.Q[obs][action] += alpha0 * (reward + gamma * max_next_q - self.Q[obs][action])
      self.Q[obs][action] = round(self.Q[obs][action], 5) #round to have less space cost to 5 decimals
      #Update the current observation to next observation
      obs = obs2
    return rewards

  #Maximize the value, we now sum the rewards from the start of the episode
  def rollout(
      self,
      env: gym.Env,
      policy,
      gamma: float,
      n_episodes: int,
      render=[],
      rewards=[],
      takeVideo=False
  ) -> float:
    """

    Args:
      env: the environment
      policy: epsilon greedy
      gamma: 0.99 usually
      n_episodes:
      max_steps: not used here
      render: true or false,

    Returns:
      collected reward

    """

    sum_returns = 0.0
    obs, info = env.reset()
    discounting = 1

    #Episodes Loop
    for ep in tqdm(range(n_episodes)):
      done = False
      obs, info = env.reset()
      discounting = 1

      while not done:
        obs = bytes(obs)
        action = self.policy(obs, self.eps, self.Q, env)
        obs, reward, terminated, truncated, info= env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        sum_returns += reward * discounting
        if takeVideo:
          frame = env.render()
          render.append(frame)
        discounting *= gamma

    print("Info: ",info)

    return sum_returns / n_episodes
#Manual saving line by line allows us to not waste RAM so that everything can go into the Qtable computation in qlearn
def save_q_table(Q, file_path):
    with open(file_path, 'w') as f:
        for state, values in Q.items():
            state_str = ','.join(map(str, state))
            values_str = ','.join(map(str, values))
            f.write(f"{state_str}:{values_str}\n")

def load_q_table_from(env, file_path):
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    with open(file_path, 'r') as f:
        for line in f:
            state_str, values_str = line.strip().split(':')
            state = bytes(map(int, state_str.split(',')))
            values = np.array(list(map(float, values_str.split(','))))
            q_table[state] = values
    return q_table

def get_reward_evolution(rewards):
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

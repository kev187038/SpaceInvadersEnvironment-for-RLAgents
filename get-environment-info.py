import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import sys

N = 1 
env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array')
env.reset()
# Run the simulation, we use python and images and mathplot to get images in real time of the environment
frames = []
rew = []
for _ in range(N):
    action = env.action_space.sample()
    obs, reward, truncated, terminated, info = env.step(action)
    print("Obs is: ",obs)
    print("Obs len is: ", len(obs))
    print("Reward is: ",reward)
    print("Truncated: ", truncated)
    print("Terminated: ", terminated)
    print("Info: ", info)
    if reward != 0:
      rew.append(reward)
    frame = env.render()
    frames.append(frame)

print("Observation space is: ", env.observation_space.shape)
print("Action space is: ", env.action_space)
print("Action space dimension is: ", env.action_space.n)
print("Action data structure is: ", env.action_space.shape)  
print("Action is: ", env.action_space.sample())  
print("Collected rewards are: ", rew)
print("State dimension is: ", sys.getsizeof(obs))
print("In Tuple representation: ", sys.getsizeof(tuple(obs)))
print("In bytes array representation: ", sys.getsizeof(bytes(obs)))
clip = ImageSequenceClip(frames, fps=20)
clip.write_videofile('./environment.mp4')
env.close()

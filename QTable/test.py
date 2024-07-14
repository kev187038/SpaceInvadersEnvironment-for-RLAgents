import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from IPython.display import HTML
from QTable import QTable
from QTable import load_q_table_from
from QTable import EpsGreedyPolicy

env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array', mode=2)
# Load the Q-table balanced bersion
table = QTable(env, eps=0.0)
table.Q = load_q_table_from(env, './q_table_space_invaders.txt')

#Create video of the Agent playing on game
frames = []
video_path = './Q-Table.mp4'
rew = table.rollout(env=env, policy=EpsGreedyPolicy(table.Q), gamma=0.99, n_episodes=5, max_steps=200, render=frames, takeVideo=True)
clip = ImageSequenceClip(frames, fps=20)
clip.write_videofile(video_path)

import gymnasium as gym
from moviepy.editor import ImageSequenceClip
from DQN import DQNAgent
from DQN import test_dqn
#Load DDQN AGENT
env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array', mode=2)
state_shape = (128, 1)
num_actions = 6
agent = DQNAgent(state_shape, num_actions, buffer_size=100000, batch_size=32, gamma=0.99, alpha=0.1,
                 epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01)
agent.load('./DQN_TM_Performance/dqn_agent.pth')

frames = []
rew = test_dqn(env=env, agent=agent, episodes=5, render=frames)
clip = ImageSequenceClip(frames, fps=20)
print("Collected reward is: ", rew)
clip.write_videofile('./output.mp4') 

import gymnasium as gym
import matplotlib.pyplot as plt
from QTable import QTable
from QTable import save_q_table
from QTable import get_reward_evolution

env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array')
env.reset()
table = QTable(env=env)
rewards = table.qlearn(env=env, alpha0=0.1, gamma=0.99, max_steps=25000000)
avgs = get_reward_evolution(rewards)
plt.plot(rewards, label='Total Reward')
plt.plot(avgs, label='Average Reward', linestyle='--')
plt.xlabel('Run')
plt.ylabel('Total Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./Rewards_evolution_Q-Table.png')
plt.show()


plt.plot(avgs, label='Avg Reward')
plt.xlabel('Run')
plt.ylabel('Avg Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./AVG_Rewards_evolution_Q-Table.png')
plt.show()


save_q_table(table.Q, './q_table_space_invaders.txt')

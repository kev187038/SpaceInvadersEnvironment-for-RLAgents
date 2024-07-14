import gymnasium as gym
import matplotlib.pyplot as plt
from DoubleDQN import DDQNAgent
from DQN import train_dqn
from DQN import get_reward_evolution

env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array')
state_shape = (128, 1)
num_actions = 6

DDQNagent = DDQNAgent(
    state_shape=state_shape,
    num_actions=num_actions,
    alpha=0.1,
    buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01
)

tup = train_dqn(env, DDQNagent, episodes=1000)
rewards = tup[0]
losses = tup[1]
actions = tup[2]
avgs = get_reward_evolution(rewards)
plt.plot(rewards, label='Total Reward')
plt.plot(avgs, label='Average Reward', linestyle='--')
plt.xlabel('Run')
plt.ylabel('Total Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./Double_DQN_Performance/Rewards_evolution_DoubleDQN.png')
plt.show()
agent.save("./Double_DQN_Performance/double_dqn_agent.pth")

plt.plot(avgs, label='Avg Reward')
plt.xlabel('Run')
plt.ylabel('Avg Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./Double_DQN_Performance/AVG_Rewards_evolution_DoubleDQN.png')
plt.show()

plt.plot(losses, label='Avg loss')
plt.xlabel('Run')
plt.ylabel('Avg loss')
plt.title('Loss evolution over episodes')
plt.savefig('./Double_DQN_Performance/Losses_evolution_DoubleDQN.png')
plt.show()

plt.hist(actions, bins=range(7), edgecolor='black', align='left')
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Histogram of Actions')
action_names = ['Noop', 'Fire', 'Move right', 'Move left', 'Rigthfire', 'Leftfire']
plt.xticks(range(6), action_names)  # Set the ticks to be the action names instead of numbers
plt.savefig('./Double_DQN_Performance/Actions_histogram_DoubleDQN.png')
plt.show()

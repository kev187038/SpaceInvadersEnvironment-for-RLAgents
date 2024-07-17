import gymnasium as gym
import matplotlib.pyplot as plt
from DQN import DQNAgent
from DQN import train_dqn
from DQN import get_reward_evolution

env = gym.make('ALE/SpaceInvaders-ram-v5', render_mode='rgb_array')
state_shape = (128, 1)
num_actions = 6

#The agent seems to work better for smaller frames: tradeoff window size / frame bound ?
agent = DQNAgent(
    state_shape=state_shape,
    num_actions=num_actions,
    alpha=0.1,
    buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
    use_target_model=False #Change this to true if you want to train a target-model DQN (DQN-TM)
)

tup = train_dqn(env, agent, episodes=1000)

#Get the data to plot
rewards = tup[0]
losses = tup[1]
actions = tup[2]

avgs = get_reward_evolution(rewards)
plt.plot(rewards, label='Total Reward')
plt.plot(avgs, label='Average Reward', linestyle='--')
plt.xlabel('Run')
plt.ylabel('Total Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./Rewards_evolution_DQN.png')
plt.show()
agent.save("./dqn_agent.pth")

plt.plot(avgs, label='Avg Reward')
plt.xlabel('Run')
plt.ylabel('Avg Reward')
plt.title('Reward Evolution Over Episodes')
plt.savefig('./AVG_Rewards_evolution_DQN.png')
plt.show()

plt.plot(losses, label='Avg loss')
plt.xlabel('Run')
plt.ylabel('Avg loss')
plt.title('Loss evolution over episodes')
plt.savefig('./Losses_evolution_DQN.png')
plt.show()


plt.hist(actions, bins=range(7), edgecolor='black', align='left')  # range(7) to cover 0 to 6 (exclusive), align to center the bins
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.title('Histogram of Actions')
action_names = ['Noop', 'Fire', 'Move right', 'Move left', 'Rigthfire', 'Leftfire']
plt.xticks(range(6), action_names)  # Set the ticks to be the action names instead of numbers
plt.savefig('drive/MyDrive/RL_Project_Data/Actions_histogram_DQN.png')
plt.show()

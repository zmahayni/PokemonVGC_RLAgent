import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

obs, info = env.reset()

print(f'Starting obs: {obs}')

episode_over=False
total_reward = 0

while not episode_over:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f'Episode Over. Total Reward: {total_reward}')
env.close()
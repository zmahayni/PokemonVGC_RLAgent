import torch
import gymnasium as gym
from src.agents.dqn_toy import DQN
from src.env.toy_env import ToyPlayer
from src.agents.experience_replay import ReplayMemory
import itertools
import yaml
import random
import matplotlib.pyplot as plt

device = "mps" if torch.mps.is_available() else "cpu"


class Toy_Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self._batch_size = hyperparameters["batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]

    def run(self, is_training=True, render=False):
        env = ToyPlayer()

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_ep = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(device=device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

        for episode in range(10000):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0.0
            while True:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append(
                        (state, action, new_state, reward, terminated, truncated)
                    )

                state = new_state

                if terminated or truncated:
                    break

            reward_per_ep.append(episode_reward)
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
            print(f'Reward = {episode_reward}, Epislon = {epsilon}')
        plt.plot(reward_per_ep)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards over Training")
        plt.show()

if __name__ == '__main__':
    agent = Toy_Agent('toy-env')
    agent.run(is_training=True, render=False)
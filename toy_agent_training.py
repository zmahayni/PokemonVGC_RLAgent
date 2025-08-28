import torch
import gymnasium as gym
from dqn_toy import DQN
from toy_env import ToyPlayer
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import os, time

device = "mps" if torch.mps.is_available() else "cpu"
run_id = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH_BEST = os.path.join(OUT_DIR, f"{run_id}_policy_best.pt")
MODEL_PATH_LAST = os.path.join(OUT_DIR, f"{run_id}_policy_last.pt")
CURVE_PATH      = os.path.join(OUT_DIR, f"{run_id}_reward_curve.png")


class Toy_Agent:
    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.batch_size = hyperparameters["batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

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

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for episode in range(10000):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0.0
            while True:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append(
                        (state, action, new_state, reward, terminated)
                    )

                    step_count += 1

                state = new_state

                if terminated:
                    break
            
            

            reward_per_ep.append(episode_reward)
            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            window = 100
            tail = reward_per_ep[-window:]
            mean_reward = sum(tail)/len(tail)
            
# keep best
            if episode == 0:
                best_mean = mean_reward
            else:
                best_mean = max(best_mean, mean_reward)
            if mean_reward >= best_mean:
                torch.save(policy_dqn.state_dict(), MODEL_PATH_BEST)
                best_mean = mean_reward
            epsilon_history.append(epsilon)

            if len(memory) > self.batch_size:
                batch = memory.sample(self.batch_size)

                self.optimize(batch, policy_dqn, target_dqn)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
        torch.save(policy_dqn.state_dict(), MODEL_PATH_LAST)
        print(f"Saved last model to {MODEL_PATH_LAST}")
        print(f"Best model path: {MODEL_PATH_BEST}")

        plt.figure()
        plt.plot(reward_per_ep, alpha=0.4, label="reward/episode")

        # moving average
        window = min(100, len(reward_per_ep))
        if window > 1:
            ma = np.convolve(reward_per_ep, np.ones(window)/window, mode="valid")
            plt.plot(range(window-1, len(reward_per_ep)), ma, label=f"{window}-ep moving avg")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(CURVE_PATH, dpi=150)
        plt.close()

        print(f"Saved curve to {CURVE_PATH}")



    def optimize(self, batch, policy_dqn, target_dqn):
                
            states, actions, new_states, rewards, terminations = zip(*batch)

            states = torch.stack(states)
            new_states = torch.stack(new_states)         
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            terminations = torch.tensor(terminations).float().to(device)

            with torch.no_grad():
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

            current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
            
            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    agent = Toy_Agent('toy-env')
    agent.run(is_training=True, render=False)
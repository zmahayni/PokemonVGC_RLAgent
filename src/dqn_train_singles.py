from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.baselines import RandomPlayer
from src.envs.Singles_Env_v1 import Singles_Env_v1
from src.dqn_toy import DQN
import torch
from src.agents.experience_replay import ReplayMemory
import yaml
from torch import nn
import random
import numpy as np
import logging
import os
import time
import matplotlib
matplotlib.use("Agg")  # Safe for headless runs; remove if you want interactive plots
import matplotlib.pyplot as plt

# Output directory and files
run_id = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)

REWARD_CURVE_PATH = os.path.join(OUT_DIR, f"{run_id}_reward_curve.png")
WINRATE_CURVE_PATH = os.path.join(OUT_DIR, f"{run_id}_winrate_curve.png")
EPS_CURVE_PATH = os.path.join(OUT_DIR, f"{run_id}_epsilon_curve.png")

logging.basicConfig(level=logging.ERROR)


device = "mps" if torch.mps.is_available() else "cpu"

def evaluate(env, policy_dqn, n_episodes=10, max_steps=500, device="cpu"):
    policy_dqn.eval()
    total_return = 0.0
    wins = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_idx = policy_dqn(state_t).squeeze().argmax().item()

            obs, reward, terminated, truncated, _ = env.step(np.int64(action_idx))
            ep_return += reward
            steps += 1

            if terminated or truncated or steps >= max_steps:
                win = bool(env.env.battle1 and env.env.battle1.won)
                wins += 1 if win else 0
                break

        total_return += ep_return

    avg_return = total_return / n_episodes
    winrate = wins / n_episodes
    policy_dqn.train()
    return avg_return, winrate

class DQN_Agent:
    def __init__(self, hyperparameter_set):
        with open("configs/hyperparameters.yml", "r") as file:
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
        agent = Singles_Env_v1()
        opponent = RandomPlayer()
        env = SingleAgentWrapper(agent, opponent)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_per_ep = []
        wins = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(device=device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(), lr=self.learning_rate_a
            )
        for i in range(10000):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0.0
            steps = 0
            while True:
                if is_training and random.random() < epsilon:
                    # action = env.action_space.sample()
                    action_idx = np.random.randint(0, 10, dtype=np.int64)
                    action = torch.tensor(action_idx, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
                steps += 1
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1

                state = new_state

                if terminated or truncated or steps >= 300:
                    break
            win = bool(env.env.battle1 and env.env.battle1.won)
            wins.append(1 if win else 0)
            reward_per_ep.append(episode_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(memory) > self.batch_size:
                batch = memory.sample(self.batch_size)

                self.optimize(batch, policy_dqn, target_dqn)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0


        eval_avg_return, eval_winrate = evaluate(env, policy_dqn, n_episodes=100, max_steps=500, device=device)
        print(f"[EVAL] Final: avg_return={eval_avg_return:.3f} | winrate={eval_winrate:.2%}")

        # 1) Reward curve with moving average
        plt.figure()
        plt.plot(reward_per_ep, alpha=0.35, label="reward/episode")
        # moving average (window = min(100, len(reward_per_ep)))
        window = min(100, len(reward_per_ep))
        if window > 1:
            ma = np.convolve(reward_per_ep, np.ones(window) / window, mode="valid")
            plt.plot(range(window - 1, len(reward_per_ep)), ma, label=f"{window}-ep moving avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REWARD_CURVE_PATH, dpi=150)
        plt.close()
        print(f"Saved reward curve to {REWARD_CURVE_PATH}")

        # 2) Win-rate curve (cumulative and rolling)
        plt.figure()
        # cumulative win-rate
        cum_winrate = np.cumsum(wins) / (np.arange(len(wins)) + 1)
        plt.plot(cum_winrate, label="Cumulative win-rate", alpha=0.8)

        # rolling win-rate over last K episodes
        K = min(100, len(wins))
        if K > 1:
            # rolling window mean using convolution
            wins_np = np.array(wins, dtype=np.float32)
            roll = np.convolve(wins_np, np.ones(K) / K, mode="valid")
            plt.plot(range(K - 1, len(wins)), roll, label=f"Rolling-{K} win-rate", alpha=0.8)

        plt.xlabel("Episode")
        plt.ylabel("Win-rate")
        plt.title("Win-rate Curve")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(WINRATE_CURVE_PATH, dpi=150)
        plt.close()
        print(f"Saved win-rate curve to {WINRATE_CURVE_PATH}")

        # 3) Epsilon curve (optional)
        plt.figure()
        plt.plot(epsilon_history, label="epsilon", color="tab:orange")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Schedule")
        plt.tight_layout()
        plt.savefig(EPS_CURVE_PATH, dpi=150)
        plt.close()
        print(f"Saved epsilon curve to {EPS_CURVE_PATH}")

    def optimize(self, batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*batch)

        states = torch.stack(states)
        new_states = torch.stack(new_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.discount_factor_g
                * target_dqn(new_states).max(dim=1)[0]
            )

        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    agent = DQN_Agent("dqn-singles-env")
    agent.run(is_training=True, render=False)


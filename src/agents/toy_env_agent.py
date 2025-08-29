import gymnasium as gym
from experiments.toy_agent_training import Toy_Agent
from src.envs.toy_env import ToyPlayer
from src.agents.experience_replay import ReplayMemory
from src.agents.dqn_toy import DQN
import torch
import poke

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "policies/toy_env_best_policy.pt"
EPISODES = 20

env = ToyPlayer()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = DQN(state_dim, action_dim).to(device)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy.eval()

def greedy_action(state):
    with torch.no_grad():
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy(s)
        return int(q.argmax(dim=1).item())
total = 0.0
for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        a = greedy_action(obs)
        obs, r, term, trunc, _ = env.step(a)
        ep_ret += r
        done = term or trunc
    print(f"Episode {ep+1}: reward={ep_ret:.3f}")
    total += ep_ret

print(f"\nAvg reward over {EPISODES} eps: {total/EPISODES:.3f}")
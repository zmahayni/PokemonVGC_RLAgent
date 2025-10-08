import os
import glob
import yaml
import numpy as np
from typing import Tuple, Optional

from stable_baselines3 import DQN
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.player.baselines import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from src.envs.Singles_Env_v1 import Singles_Env_v1
from src.sb3_action_mappingSingles import (
    build_mask_from_battle,
    DEFAULT_RESERVED_INDEX,
    mask_fn,
)


# DQN evaluation wrapper: mirrors training-time handling to avoid invalid actions
class MaskedSingleAgentWrapper(SingleAgentWrapper):
    def step(self, action):
        poke_env = self.env  # underlying Singles_Env_v1
        act_size = self.action_space.n
        battle = getattr(poke_env, "battle1", None)

        # Build mask for current timestep
        mask = build_mask_from_battle(battle=battle, act_size=act_size)

        # Translate reserved default index to -2 only if default is the sole valid order
        vo = None if battle is None else getattr(battle, "valid_orders", None)
        if (
            vo is not None
            and isinstance(vo, (list, tuple))
            and len(vo) == 1
            and str(vo[0]).strip() == "/choose default"
            and int(action) == int(DEFAULT_RESERVED_INDEX)
        ):
            action = np.int64(-2)
        else:
            # If chosen action is invalid, pick a valid one
            try:
                a = int(action)
            except Exception:
                a = None
            if a is None or a < 0 or a >= act_size or not mask[a]:
                valid_actions = [i for i, ok in enumerate(mask) if ok]
                if valid_actions:
                    # keep as numpy scalar to satisfy downstream expectations
                    action = np.int64(np.random.choice(valid_actions))

        # Ensure the env receives a numpy scalar (has .item())
        action = np.int64(action)

        return super().step(action)


def _latest_final_model(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    # find subdirs that contain final_model.zip and take the most recent by name
    candidates: list[Tuple[str, str]] = []  # (subdir, model_path)
    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if not os.path.isdir(subdir):
            continue
        model_path = os.path.join(subdir, "final_model.zip")
        if os.path.isfile(model_path):
            candidates.append((subdir, model_path))
    if not candidates:
        # also try deeper search in case final_model.zip is nested differently
        deep = glob.glob(
            os.path.join(base_dir, "**", "final_model.zip"), recursive=True
        )
        if not deep:
            return None
        # choose the lexicographically latest path
        deep.sort()
        return deep[-1]
    # choose latest by subdir name (timestamps are used in run folder names)
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_teams_from_cfg(
    cfg_path: str,
    ppo_section: str = "ppo-singles-env",
    dqn_section: str = "dqn-singles-env",
):
    with open(cfg_path, "r") as f:
        cfg_all = yaml.safe_load(f)
    # Prefer PPO section for team paths; fallback to DQN if absent
    sec = cfg_all.get(ppo_section) or cfg_all.get(dqn_section) or {}
    teams_dir = sec.get("teams_dir", "teams")
    agent_team_path = sec.get(
        "agent_team_path", os.path.join(teams_dir, "agent_ou.txt")
    )
    opponent_team_path = sec.get(
        "opponent_team_path", os.path.join(teams_dir, "opponent_ou.txt")
    )
    with open(agent_team_path, "r", encoding="utf-8") as f:
        agent_team_str = f.read()
    with open(opponent_team_path, "r", encoding="utf-8") as f:
        opponent_team_str = f.read()
    return agent_team_str, opponent_team_str


def _make_env_for_ppo(agent_team_str: str, opponent_team_str: str):
    agent = Singles_Env_v1(
        battle_format="gen9ou",
        team=ConstantTeambuilder(agent_team_str),
    )
    opponent = RandomPlayer(
        battle_format="gen9ou",
        team=ConstantTeambuilder(opponent_team_str),
    )
    base_env = SingleAgentWrapper(env=agent, opponent=opponent)
    masked_env = ActionMasker(base_env, mask_fn)
    return masked_env


def _make_env_for_dqn(agent_team_str: str, opponent_team_str: str):
    agent = Singles_Env_v1(
        battle_format="gen9ou",
        team=ConstantTeambuilder(agent_team_str),
    )
    opponent = RandomPlayer(
        battle_format="gen9ou",
        team=ConstantTeambuilder(opponent_team_str),
    )
    base_env = MaskedSingleAgentWrapper(env=agent, opponent=opponent)
    return base_env


def _run_episodes(model, env, n_episodes: int = 100, algo: str = "PPO"):
    rewards = []
    wins = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        steps = 0
        while not done:
            mask = None
            if algo.upper() == "PPO":
                mask = mask_fn(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_rew += float(reward)
            steps += 1

        # Determine win from underlying battle object
        try:
            if algo.upper() == "PPO":
                # env -> ActionMasker -> SingleAgentWrapper -> Singles_Env_v1
                battle = getattr(getattr(env.env, "env", None), "battle1", None)
            else:
                # env -> MaskedSingleAgentWrapper -> Singles_Env_v1
                battle = getattr(env.env, "battle1", None)
            won = bool(getattr(battle, "won", False))
        except Exception:
            won = False

        wins += int(won)
        rewards.append(ep_rew)
        print(f"[{algo}] Episode {ep:03d}: reward={ep_rew:.3f} win={won}")

    winrate = 100.0 * wins / max(1, n_episodes)
    avg_rew = float(np.mean(rewards)) if rewards else 0.0

    print("=" * 60)
    print(f"{algo} vs RandomPlayer over {n_episodes} episodes")
    print(f"Winrate: {winrate:.2f}%  |  Average reward: {avg_rew:.3f}")
    print("=" * 60)

    return rewards, wins, winrate, avg_rew


def main():
    # Discover latest final models
    ppo_path = _latest_final_model("runs/ppo_singles")
    dqn_path = _latest_final_model("runs/dqn_singles")

    if ppo_path is None and dqn_path is None:
        raise FileNotFoundError(
            "No final_model.zip found in runs/ppo_singles or runs/dqn_singles"
        )

    # Load teams
    agent_team_str, opponent_team_str = _load_teams_from_cfg(
        "configs/hyperparameters.yml"
    )

    # Evaluate PPO
    if ppo_path is not None:
        env_ppo = _make_env_for_ppo(agent_team_str, opponent_team_str)
        ppo_model = MaskablePPO.load(ppo_path, env=env_ppo, device="auto")
        _run_episodes(ppo_model, env_ppo, n_episodes=1000, algo="PPO")

    # Evaluate DQN
    if dqn_path is not None:
        env_dqn = _make_env_for_dqn(agent_team_str, opponent_team_str)
        dqn_model = DQN.load(dqn_path, env=env_dqn, device="auto")
        _run_episodes(dqn_model, env_dqn, n_episodes=1000, algo="DQN")


if __name__ == "__main__":
    main()

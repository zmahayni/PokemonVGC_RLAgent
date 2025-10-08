import os
import time
import yaml
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.Singles_Env_v1 import Singles_Env_v1
from poke_env.player.baselines import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from src.sb3_action_mappingSingles import (
    build_mask_from_battle,
    DEFAULT_RESERVED_INDEX,
)


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

if __name__ == "__main__":
    # 1) Load config
    with open("configs/hyperparameters.yml", "r") as f:
        cfg_all = yaml.safe_load(f)
        cfg = cfg_all.get("dqn-singles-env", {})

    # 2) Paths and run id
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(cfg.get("tensorboard_log_dir", "runs/dqn_singles"), run_id)
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # 3) Common settings
    total_timesteps = int(cfg.get("total_timesteps", 1_000_000))
    eval_episodes = int(cfg.get("eval_episodes", 50))
    eval_freq_steps = int(cfg.get("eval_freq_steps", 10_000))
    n_envs = int(cfg.get("n_envs", 1))
    seed = int(cfg.get("seed", 0))

    # DQN-specific settings (mapped from your existing names)
    buffer_size = int(cfg.get("replay_memory_size", 100_000))
    batch_size = int(cfg.get("batch_size", 64))
    learning_rate = float(cfg.get("learning_rate_a", 1e-3))
    gamma = float(cfg.get("discount_factor_g", 0.99))
    target_update_interval = int(cfg.get("network_sync_rate", 500))

    # Exploration schedule mapping
    exploration_initial_eps = float(cfg.get("epsilon_init", 1.0))
    exploration_final_eps = float(cfg.get("epsilon_min", 0.05))
    # SB3 uses exploration_fraction to anneal eps over the first fraction of training steps.
    # If not provided, use a reasonable default.
    exploration_fraction = float(cfg.get("exploration_fraction", 0.5))

    # Network
    # If you add `net_arch: [256, 256]` under dqn-singles-env in the YAML, it will be used here.
    net_arch = cfg.get("net_arch", [256, 256])
    policy_kwargs = {"net_arch": net_arch}

    # Load OU teams (Showdown export text files)
    teams_dir = cfg.get("teams_dir", "teams")
    agent_team_path = cfg.get("agent_team_path", os.path.join(teams_dir, "agent_ou.txt"))
    opponent_team_path = cfg.get("opponent_team_path", os.path.join(teams_dir, "opponent_ou.txt"))
    with open(agent_team_path, "r", encoding="utf-8") as f:
        agent_team_str = f.read()
    with open(opponent_team_path, "r", encoding="utf-8") as f:
        opponent_team_str = f.read()

    # 4) Build train env (DummyVecEnv with a factory callable) - NO masking
    def make_env():
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

    train_env = make_vec_env(
        make_env, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv
    )
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 10, vec_env_cls=DummyVecEnv)

    # 5) Build DQN model
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=out_dir,
        seed=seed,
        device="auto",
        verbose=1,
    )

    # 6) Evaluation callback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=out_dir,
        eval_freq=eval_freq_steps,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    # 7) Train and save
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb], progress_bar=True)

    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)

    # 8) Cleanup
    train_env.close()
    eval_env.close()

    print(
        f"Training complete. Final model: {final_path}\n"
        f"Best model dir: {best_dir}\n"
        f"TensorBoard logs: {out_dir}"
    )

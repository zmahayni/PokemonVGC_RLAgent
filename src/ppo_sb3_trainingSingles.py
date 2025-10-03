import os
import time
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from src.envs.Singles_Env_v1 import Singles_Env_v1
from poke_env.player.baselines import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from gymnasium import Env
from typing import Any, Dict, Tuple
from gymnasium.spaces import Discrete


def map_orders_to_indices(valid_orders, battle):
    indices = []
    for order in valid_orders:
        idx_np = SinglesEnv.order_to_action(
            order = order,
            battle = battle,
            fake = True,
            strict = False
        )
        idx = int(idx_np)
        indices.append(idx)
    return indices

def build_mask_from_battle(battle, act_size):
    """
    Build a boolean mask (np.ndarray of dtype=bool, shape [act_size]) where True means
    the action index is legal at this timestep.

    Fallback: if no valid_orders are present (e.g., between episodes), return all True.
    """
    # Default: mask off everything
    mask = np.zeros(act_size, dtype=bool)

    # Between episodes or not ready: allow all to avoid crashes
    if battle is None or battle.valid_orders is None or len(battle.valid_orders) == 0:
        return np.ones(act_size, dtype=bool)

    # Map valid orders to action indices and set those to True
    indices = map_orders_to_indices(valid_orders=battle.valid_orders, battle=battle)
    for idx in indices:
        if 0 <= idx < act_size:
            mask[idx] = True

    # Safety: ensure at least one action is allowed
    if not mask.any():
        mask[:] = True

    return mask

    
def mask_fn(env):
    base_env = env
    poke_env = base_env.env
    act_size = base_env.action_space.n
    battle = poke_env.battle1
    return build_mask_from_battle(battle=battle, act_size=act_size)
            

if __name__ == "__main__":
    # 1) Load config
    with open("configs/hyperparameters.yml", "r") as f:
        cfg = yaml.safe_load(f)["ppo-singles-env"]

    # 2) Paths and run id
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(cfg.get("tensorboard_log_dir", "runs/ppo_singles"), run_id)
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # 3) Common settings
    total_timesteps = int(cfg.get("total_timesteps", 1_000_000))
    eval_episodes = int(cfg.get("eval_episodes", 100))
    eval_freq_steps = int(cfg.get("eval_freq_steps", 10_000))
    n_envs = int(cfg.get("n_envs", 1))
    seed = int(cfg.get("seed", 0))

    # 4) Build train env (DummyVecEnv with a factory callable)
    def make_env():
        agent = Singles_Env_v1()
        opponent = RandomPlayer()
        base_env = SingleAgentWrapper(env=agent, opponent=opponent)
        masked_env = ActionMasker(base_env, mask_fn)

        return masked_env
        
    train_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    eval_env  = make_vec_env(make_env, n_envs=1,     seed=seed+10, vec_env_cls=DummyVecEnv)

    # 6) Policy kwargs (activation + net_arch)
    act = str(cfg.get("activation_fn", "relu")).lower()
    if act == "tanh":
        from torch.nn import Tanh as ACT
    else:
        from torch.nn import ReLU as ACT  # default

    policy_kwargs = {
        "net_arch": cfg.get("net_arch", {"pi": [128, 128], "vf": [128, 128]}),
        "activation_fn": ACT,
    }

    # 7) Build PPO model
    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 1024)),
        batch_size=int(cfg.get("batch_size", 256)),
        n_epochs=int(cfg.get("n_epochs", 10)),
        gamma=float(cfg.get("gamma", 0.995)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        target_kl=cfg.get("target_kl", 0.02),
        policy_kwargs=policy_kwargs,
        tensorboard_log=out_dir,
        seed=seed,
        device="auto",
        verbose=1,
    )

    # 8) Evaluation callback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=out_dir,
        eval_freq=eval_freq_steps,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    # 9) Train and save
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb], progress_bar=True)

    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)

    # 10) Cleanup
    train_env.close()
    eval_env.close()

    print(
        f"Training complete. Final model: {final_path}\n"
        f"Best model dir: {best_dir}\n"
        f"TensorBoard logs: {out_dir}"
    )

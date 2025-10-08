import os
import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.Doubles_Env_v1 import Doubles_Env_v1
from poke_env.player.baselines import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

# Note: sb3-contrib MaskablePPO does not support MultiDiscrete action spaces used by DoublesEnv.
# This boilerplate uses standard PPO (SB3) without action masking.

if __name__ == "__main__":
    # 1) Load config
    with open("configs/hyperparameters.yml", "r") as f:
        cfg = yaml.safe_load(f).get("ppo-doubles-env", {})

    # 2) Paths and run id
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(cfg.get("tensorboard_log_dir", "runs/ppo_doubles"), run_id)
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # 3) Common settings
    total_timesteps = int(cfg.get("total_timesteps", 1_000_000))
    eval_episodes = int(cfg.get("eval_episodes", 50))
    eval_freq_steps = int(cfg.get("eval_freq_steps", 10_000))
    n_envs = int(cfg.get("n_envs", 1))
    seed = int(cfg.get("seed", 0))

    # Teams
    teams_dir = cfg.get("teams_dir", "teams")
    agent_team_path = cfg.get("agent_team_path", os.path.join(teams_dir, "doubles_agent_regI.txt"))
    opponent_team_path = cfg.get("opponent_team_path", os.path.join(teams_dir, "doubles_opp_regI.txt"))
    with open(agent_team_path, "r", encoding="utf-8") as f:
        agent_team_str = f.read()
    with open(opponent_team_path, "r", encoding="utf-8") as f:
        opponent_team_str = f.read()

    # 4) Build train env
    def make_env():
        agent = Doubles_Env_v1(
            battle_format=cfg.get("battle_format", "gen9vgc2025regi"),
            team=ConstantTeambuilder(agent_team_str),
        )
        opponent = RandomPlayer(
            battle_format=cfg.get("battle_format", "gen9vgc2025regi"),
            team=ConstantTeambuilder(opponent_team_str),
        )
        env = SingleAgentWrapper(env=agent, opponent=opponent)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 10, vec_env_cls=DummyVecEnv)

    # 5) Policy kwargs (activation + net_arch)
    act = str(cfg.get("activation_fn", "relu")).lower()
    if act == "tanh":
        from torch.nn import Tanh as ACT
    else:
        from torch.nn import ReLU as ACT  # default

    policy_kwargs = {
        "net_arch": cfg.get("net_arch", {"pi": [256, 256], "vf": [256, 256]}),
        "activation_fn": ACT,
    }

    # 6) Build PPO model (standard SB3 PPO supports MultiDiscrete; no masking)
    model = PPO(
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

    # 7) Evaluation callback
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=out_dir,
        eval_freq=eval_freq_steps,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    # 8) Train and save
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb], progress_bar=True)

    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)

    # 9) Cleanup
    train_env.close()
    eval_env.close()

    print(
        f"Training complete. Final model: {final_path}\n"
        f"Best model dir: {best_dir}\n"
        f"TensorBoard logs: {out_dir}"
    )

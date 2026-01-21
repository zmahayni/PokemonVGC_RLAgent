"""
Train VGC agent against a heuristic opponent with a different team.

The heuristic opponent chooses moves based on damage estimation:
- Base power
- Type effectiveness
- STAB bonus
- Physical vs Special stat ratios

Prerequisites:
    1. Start Pokemon Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security

    2. Create opponent team file at teams/team_opponent.txt

Usage:
    python src/training/train_vs_heuristic.py

The trained model will be saved to models/vgc_v2_heuristic_*.zip
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Disable PyTorch distribution validation to avoid numerical precision errors
torch.distributions.Distribution.set_default_validate_args(False)

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "poke-env" / "src"))

from src.envs import VGCEnvV2


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that preserves the action_masks() method for MaskablePPO.
    """

    _debug_counter = 0

    def action_masks(self):
        """Forward action_masks() to the underlying VGCEnv."""
        mask = self.unwrapped.action_masks()
        ActionMaskWrapper._debug_counter += 1
        if ActionMaskWrapper._debug_counter <= 3 or ActionMaskWrapper._debug_counter % 500 == 0:
            num_valid = mask.sum()
            print(f"[Mask] Call #{ActionMaskWrapper._debug_counter}: {num_valid} valid", flush=True)
        return mask


# Import MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    HAS_MASKABLE_PPO = True
except ImportError:
    print("Error: sb3-contrib not installed. Install with: pip install sb3-contrib")
    print("MaskablePPO is required for training.")
    sys.exit(1)


# Configuration
CONFIG = {
    "battle_format": "gen9vgc2026regf",
    "team_path": PROJECT_ROOT / "teams" / "team.txt",
    "opponent_team_path": PROJECT_ROOT / "teams" / "team_opponent.txt",
    "opponent_type": "heuristic",  # "random", "heuristic", or "max_power"
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "save_freq": 50_000,
    "log_dir": PROJECT_ROOT / "logs",
    "model_dir": PROJECT_ROOT / "models",
    # Set to a model path to resume training, or None to start fresh
    "resume_from": PROJECT_ROOT / "models" / "vgc_v2_heuristic_20260119_110615_final.zip",
}


class ProgressCallback(BaseCallback):
    """Custom callback with progress bar and battle statistics."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.n_battles = 0
        self.n_wins = 0
        self.recent_rewards = []

    def _on_training_start(self):
        desc = "Continuing vs Heuristic" if CONFIG.get("resume_from") else "Training vs Heuristic"
        self.pbar = tqdm(total=self.total_timesteps, desc=desc, unit="steps")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)

        for info in self.locals.get("infos", []):
            if "won" in info and info["won"] is not None:
                self.n_battles += 1
                if info["won"]:
                    self.n_wins += 1

                if "episode" in info:
                    self.recent_rewards.append(info["episode"]["r"])
                    if len(self.recent_rewards) > 100:
                        self.recent_rewards.pop(0)

                win_rate = self.n_wins / self.n_battles if self.n_battles > 0 else 0
                avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
                self.pbar.set_postfix({
                    "battles": self.n_battles,
                    "win_rate": f"{win_rate:.1%}",
                    "avg_reward": f"{avg_reward:.2f}"
                })

        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

        win_rate = self.n_wins / self.n_battles if self.n_battles > 0 else 0
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total battles: {self.n_battles}")
        print(f"  Wins: {self.n_wins}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"{'='*60}")


def train():
    """Main training function for V2 environment with heuristic opponent."""
    print("=" * 60)
    print("VGC RL Agent Training - vs Heuristic Opponent")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Opponent type: {CONFIG['opponent_type']}")
    print(f"  Agent team: {CONFIG['team_path']}")
    print(f"  Opponent team: {CONFIG['opponent_team_path']}")
    print()

    # Check opponent team exists
    if not CONFIG["opponent_team_path"].exists():
        print(f"Error: Opponent team file not found at {CONFIG['opponent_team_path']}")
        print("Please create the opponent team file first.")
        print("\nYou can copy and modify the existing team:")
        print(f"  cp {CONFIG['team_path']} {CONFIG['opponent_team_path']}")
        return None

    # Create directories
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vgc_v2_heuristic_{timestamp}"

    print(f"  Total timesteps: {CONFIG['total_timesteps']:,}")
    if CONFIG.get("resume_from"):
        print(f"  Resuming from: {CONFIG['resume_from']}")
    print()

    print("Creating V2 environment with heuristic opponent...")
    env = VGCEnvV2(
        battle_format=CONFIG["battle_format"],
        team_path=CONFIG["team_path"],
        opponent_team_path=CONFIG["opponent_team_path"],
        opponent_type=CONFIG["opponent_type"],
    )
    # Wrap with Monitor for episode stats, then ActionMaskWrapper for action_masks()
    env = Monitor(env)
    env = ActionMaskWrapper(env)

    # Load existing model or create new one
    if CONFIG.get("resume_from") and Path(CONFIG["resume_from"]).exists():
        print(f"Loading model from {CONFIG['resume_from']}...")
        model = MaskablePPO.load(
            str(CONFIG["resume_from"]),
            env=env,
            tensorboard_log=str(CONFIG["log_dir"]),
        )
        print("Model loaded successfully!")
    else:
        print("Creating new MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=CONFIG["learning_rate"],
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=CONFIG["n_epochs"],
            gamma=CONFIG["gamma"],
            gae_lambda=CONFIG["gae_lambda"],
            clip_range=CONFIG["clip_range"],
            ent_coef=CONFIG["ent_coef"],
            vf_coef=CONFIG["vf_coef"],
            max_grad_norm=CONFIG["max_grad_norm"],
            verbose=0,
            tensorboard_log=str(CONFIG["log_dir"]),
        )

    callbacks = [
        ProgressCallback(total_timesteps=CONFIG["total_timesteps"], verbose=0),
        CheckpointCallback(
            save_freq=CONFIG["save_freq"],
            save_path=str(CONFIG["model_dir"]),
            name_prefix=run_name,
            verbose=0,
        ),
    ]

    print("\nStarting training...")
    print("(Make sure Pokemon Showdown server is running!)")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=callbacks,
            tb_log_name=run_name,
        )

        final_path = CONFIG["model_dir"] / f"{run_name}_final"
        model.save(str(final_path))
        print(f"\nTraining complete! Model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        interrupt_path = CONFIG["model_dir"] / f"{run_name}_interrupted"
        model.save(str(interrupt_path))
        print(f"Model saved to: {interrupt_path}")

    finally:
        env.close()

    return model


if __name__ == "__main__":
    train()

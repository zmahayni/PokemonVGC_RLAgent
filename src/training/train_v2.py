"""
Train VGC agent with V2 environment (improved observations and rewards).

V2 improvements over original:
- 86-dimensional observations (vs 64): stat stages, status, weather, terrain, type matchups
- Fixed bench count (2 slots for VGC format)
- Enhanced rewards: status infliction, stat boosts, speed control setup

Prerequisites:
    1. Start Pokemon Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security

    2. Install sb3-contrib (for MaskablePPO):
       pip install sb3-contrib

Usage:
    python src/training/train_v2.py

The trained model will be saved to models/vgc_v2_*.zip
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

    Monitor wrapper doesn't forward action_masks(), which breaks MaskablePPO.
    This wrapper ensures action_masks() is accessible at the top level.
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
    print("MaskablePPO is required for V2 environment training.")
    sys.exit(1)


# Configuration
CONFIG = {
    "battle_format": "gen9vgc2026regf",
    "team_path": PROJECT_ROOT / "teams" / "team.txt",
    "total_timesteps": 100_000,
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
    "save_freq": 10_000,
    "log_dir": PROJECT_ROOT / "logs",
    "model_dir": PROJECT_ROOT / "models",
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
        self.pbar = tqdm(total=self.total_timesteps, desc="Training V2", unit="steps")

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
    """Main training function for V2 environment."""
    print("=" * 60)
    print("VGC RL Agent Training - V2 Environment")
    print("=" * 60)
    print("\nV2 Improvements:")
    print("  - 86-dim observations (stat stages, status, weather, terrain, type matchups)")
    print("  - Fixed bench count (2 slots)")
    print("  - Enhanced rewards (status, boosts, speed control)")
    print()

    # Create directories
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vgc_v2_{timestamp}"

    print(f"Configuration:")
    print(f"  Format: {CONFIG['battle_format']}")
    print(f"  Team: {CONFIG['team_path']}")
    print(f"  Total timesteps: {CONFIG['total_timesteps']:,}")
    print()

    print("Creating V2 environment...")
    env = VGCEnvV2(
        battle_format=CONFIG["battle_format"],
        team_path=CONFIG["team_path"],
    )
    # Wrap with Monitor for episode stats, then ActionMaskWrapper for action_masks()
    env = Monitor(env)
    env = ActionMaskWrapper(env)

    print("Creating MaskablePPO model...")
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

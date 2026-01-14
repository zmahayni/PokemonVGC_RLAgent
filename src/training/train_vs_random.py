"""
Phase 1 Training: Train VGC agent against random opponent.

This script trains a PPO agent against a random opponent using action masking
to handle the joint-legality constraints in VGC doubles battles.

Prerequisites:
    1. Start Pokemon Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security

    2. Install sb3-contrib (for MaskablePPO):
       pip install sb3-contrib

Usage:
    python src/training/train_vs_random.py

The trained model will be saved to models/vgc_ppo_vs_random.zip
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "poke-env" / "src"))

# CRITICAL: Disable ALL logging below CRITICAL level globally
# This is the most aggressive way to suppress poke-env's verbose logging
logging.disable(logging.WARNING)

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

# Try to import MaskablePPO from sb3-contrib
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_MASKABLE_PPO = True
except ImportError:
    print("Warning: sb3-contrib not installed. Install with: pip install sb3-contrib")
    print("Falling back to standard PPO (without action masking)")
    from stable_baselines3 import PPO
    HAS_MASKABLE_PPO = False

from src.envs import VGCEnv


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
    "eval_freq": 5_000,
    "log_dir": PROJECT_ROOT / "logs",
    "model_dir": PROJECT_ROOT / "models",
}


def get_action_mask(env: VGCEnv) -> np.ndarray:
    """
    Get action mask from environment for MaskablePPO.

    Returns shape (214,) = (107 + 107) for MultiDiscrete([107, 107]).
    First 107 values are mask for position 0, next 107 for position 1.
    """
    return env.action_masks()


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
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")

    def _on_step(self) -> bool:
        # Update progress bar
        if self.pbar:
            self.pbar.update(1)

        # Check for episode end
        for info in self.locals.get("infos", []):
            if "won" in info and info["won"] is not None:
                self.n_battles += 1
                if info["won"]:
                    self.n_wins += 1

                # Track recent episode rewards
                if "episode" in info:
                    self.recent_rewards.append(info["episode"]["r"])
                    if len(self.recent_rewards) > 100:
                        self.recent_rewards.pop(0)

                # Update progress bar description
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

        # Print final stats
        win_rate = self.n_wins / self.n_battles if self.n_battles > 0 else 0
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Total battles: {self.n_battles}")
        print(f"  Wins: {self.n_wins}")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"{'='*60}")


def make_env(config: dict):
    """Create a VGC environment."""
    def _init():
        env = VGCEnv(
            battle_format=config["battle_format"],
            team_path=config["team_path"],
        )
        return Monitor(env)
    return _init


def setup_logging(log_dir: Path, run_name: str):
    """Configure logging to file only (console is disabled globally)."""
    log_file = log_dir / f"{run_name}.log"

    # Re-enable logging for file output only
    logging.disable(logging.NOTSET)  # Re-enable logging

    # Create file handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Configure root logger with file handler only (no console)
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Remove all handlers
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Disable console output by setting high level for StreamHandlers
    # that might be added later by poke-env
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.CRITICAL)

    return log_file


def train():
    """Main training function."""
    print("=" * 60)
    print("VGC RL Agent Training - Phase 1: vs Random Opponent")
    print("=" * 60)

    # Create directories
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vgc_ppo_{timestamp}"

    # Setup logging to file only BEFORE creating environment
    log_file = setup_logging(CONFIG["log_dir"], run_name)

    print(f"\nConfiguration:")
    print(f"  Format: {CONFIG['battle_format']}")
    print(f"  Team: {CONFIG['team_path']}")
    print(f"  Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"  Using MaskablePPO: {HAS_MASKABLE_PPO}")
    print(f"  Log file: {log_file}")
    print()

    # Create environment (logging already configured)
    print("Creating environment...")
    env = VGCEnv(
        battle_format=CONFIG["battle_format"],
        team_path=CONFIG["team_path"],
    )

    # Wrap with action masker if using MaskablePPO
    if HAS_MASKABLE_PPO:
        env = ActionMasker(env, get_action_mask)

    env = Monitor(env)

    # Create model (verbose=0 to reduce output)
    print("Creating model...")
    if HAS_MASKABLE_PPO:
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
    else:
        model = PPO(
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

    # Callbacks
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

        # Save final model
        final_path = CONFIG["model_dir"] / f"{run_name}_final"
        model.save(str(final_path))
        print(f"\nTraining complete! Model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        # Save interrupted model
        interrupt_path = CONFIG["model_dir"] / f"{run_name}_interrupted"
        model.save(str(interrupt_path))
        print(f"Model saved to: {interrupt_path}")

    finally:
        env.close()

    return model


if __name__ == "__main__":
    train()

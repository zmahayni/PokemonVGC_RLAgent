"""
Model vs Model Evaluation Script.

Pits two trained RL agents against each other:
- Agent 1: Discrete action space (MaskablePPO) - newer model
- Agent 2: MultiDiscrete action space (standard PPO) - older model

Prerequisites:
    1. Start Pokemon Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    python src/evaluation/model_vs_model.py
"""

import asyncio
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Disable PyTorch distribution validation to avoid numerical precision errors
# when action masking creates extreme logit values (same fix as training script)
torch.distributions.Distribution.set_default_validate_args(False)

# Suppress verbose logging
logging.getLogger("poke_env").setLevel(logging.WARNING)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "poke-env" / "src"))

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import BattleOrder

from src.players.vgc_player import load_team, random_account, DEFAULT_TEAM_PATH
from src.utils.action_masking import (
    ACTION_SPACE_SIZE,
    get_action_mask_flat,
    flat_to_action_pair,
)
from src.utils.battle_features import (
    OBS_DIM_V2,
    extract_all_features,
    features_to_observation,
)

# Try to import models
try:
    from sb3_contrib import MaskablePPO
    HAS_MASKABLE = True
except ImportError:
    HAS_MASKABLE = False

from stable_baselines3 import PPO


# Model paths
DISCRETE_MODEL_PATH = PROJECT_ROOT / "models" / "vgc_ppo_20260115_110312_final.zip"
MULTIDISCRETE_MODEL_PATH = PROJECT_ROOT / "models" / "vgc_ppo_20260114_134618_final.zip"


class DiscreteModelPlayer(Player):
    """
    Player that uses a trained MaskablePPO model with Discrete(11449) action space.
    """

    def __init__(
        self,
        model_path: Path,
        battle_format: str = "gen9vgc2026regf",
        team: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            account_configuration=random_account("Discrete"),
            battle_format=battle_format,
            team=team or load_team(),
            accept_open_team_sheet=True,
            server_configuration=LocalhostServerConfiguration,
            **kwargs
        )

        if HAS_MASKABLE:
            self.model = MaskablePPO.load(str(model_path))
        else:
            raise ImportError("MaskablePPO not available. Install sb3-contrib.")

        print(f"[DiscreteModelPlayer] Loaded model from {model_path}")

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6 for VGC)."""
        # VGC teams have 6 Pokemon, we select 4
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Use the trained model to select an action."""
        # Create observation (simplified - should match training obs)
        obs = self._embed_battle(battle)

        # Get action mask
        mask = get_action_mask_flat(battle)

        # Predict action
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)

        # Convert flat action to action pair
        a0, a1 = flat_to_action_pair(int(action))
        action_pair = np.array([a0, a1])

        # Convert to battle order
        try:
            order = DoublesEnv.action_to_order(action_pair, battle, fake=False, strict=False)
            return order
        except Exception as e:
            print(f"[DiscreteModelPlayer] Error converting action: {e}")
            return self.choose_random_doubles_move(battle)

    def _embed_battle(self, battle: DoubleBattle) -> np.ndarray:
        """Create observation embedding (must match training)."""
        OBS_DIM = 64
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        idx = 0

        # Active Pokemon HP (2)
        for i in range(2):
            if i < len(battle.active_pokemon) and battle.active_pokemon[i] is not None:
                obs[idx] = battle.active_pokemon[i].current_hp_fraction
            idx += 1

        # Bench Pokemon HP (4)
        bench = [m for m in battle.team.values() if not m.active and not m.fainted]
        for i in range(4):
            if i < len(bench):
                obs[idx] = bench[i].current_hp_fraction
            idx += 1

        # Opponent active Pokemon HP (2)
        for i in range(2):
            if i < len(battle.opponent_active_pokemon) and battle.opponent_active_pokemon[i] is not None:
                obs[idx] = battle.opponent_active_pokemon[i].current_hp_fraction
            idx += 1

        # Opponent bench HP (4)
        opp_bench = [m for m in battle.opponent_team.values() if not m.active and not m.fainted]
        for i in range(4):
            if i < len(opp_bench):
                obs[idx] = opp_bench[i].current_hp_fraction
            idx += 1

        # Gimmick availability (8)
        for i in range(2):
            obs[idx] = float(battle.can_tera[i]) if i < len(battle.can_tera) else 0.0
            idx += 1
            obs[idx] = float(battle.can_mega_evolve[i]) if i < len(battle.can_mega_evolve) else 0.0
            idx += 1
            obs[idx] = float(battle.can_dynamax[i]) if i < len(battle.can_dynamax) else 0.0
            idx += 1
            obs[idx] = float(battle.can_z_move[i]) if i < len(battle.can_z_move) else 0.0
            idx += 1

        # Force switch (2)
        for i in range(2):
            obs[idx] = float(battle.force_switch[i]) if i < len(battle.force_switch) else 0.0
            idx += 1

        # Turn number (1)
        obs[idx] = min(battle.turn / 50.0, 1.0)

        return obs


class V2ModelPlayer(Player):
    """
    Player that uses a trained MaskablePPO model with V2 environment (86-dim obs).
    """

    def __init__(
        self,
        model_path: Path,
        battle_format: str = "gen9vgc2026regf",
        team: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            account_configuration=random_account("V2Model"),
            battle_format=battle_format,
            team=team or load_team(),
            accept_open_team_sheet=True,
            server_configuration=LocalhostServerConfiguration,
            **kwargs
        )

        if HAS_MASKABLE:
            self.model = MaskablePPO.load(str(model_path))
        else:
            raise ImportError("MaskablePPO not available. Install sb3-contrib.")

        print(f"[V2ModelPlayer] Loaded model from {model_path}")

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6 for VGC)."""
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Use the trained model to select an action."""
        # Use V2 observation embedding (86 features)
        obs = self._embed_battle_v2(battle)

        # Get action mask
        mask = get_action_mask_flat(battle)

        # Predict action
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)

        # Convert flat action to action pair
        a0, a1 = flat_to_action_pair(int(action))
        action_pair = np.array([a0, a1])

        # Convert to battle order
        try:
            order = DoublesEnv.action_to_order(action_pair, battle, fake=False, strict=False)
            return order
        except Exception as e:
            print(f"[V2ModelPlayer] Error converting action: {e}")
            return self.choose_random_doubles_move(battle)

    def _embed_battle_v2(self, battle: DoubleBattle) -> np.ndarray:
        """Create V2 observation embedding (86 features)."""
        try:
            features = extract_all_features(battle)
            return features_to_observation(features)
        except Exception:
            return np.zeros(OBS_DIM_V2, dtype=np.float32)


class MultiDiscreteModelPlayer(Player):
    """
    Player that uses a trained model with MultiDiscrete([107, 107]) action space.
    """

    def __init__(
        self,
        model_path: Path,
        battle_format: str = "gen9vgc2026regf",
        team: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            account_configuration=random_account("MultiD"),
            battle_format=battle_format,
            team=team or load_team(),
            accept_open_team_sheet=True,
            server_configuration=LocalhostServerConfiguration,
            **kwargs
        )

        # Try MaskablePPO first, fall back to PPO
        try:
            if HAS_MASKABLE:
                self.model = MaskablePPO.load(str(model_path))
            else:
                self.model = PPO.load(str(model_path))
        except TypeError:
            # Model might be standard PPO
            self.model = PPO.load(str(model_path))
        print(f"[MultiDiscreteModelPlayer] Loaded model from {model_path}")

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6 for VGC)."""
        # VGC teams have 6 Pokemon, we select 4
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Use the trained model to select an action."""
        # Create observation (simplified - should match training obs)
        obs = self._embed_battle(battle)

        # Get action mask for MultiDiscrete (shape: [107, 107])
        mask = self._get_multidiscrete_mask(battle)

        # Predict action (MultiDiscrete returns [action1, action2])
        try:
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        except TypeError:
            # Model might not support action masks
            action, _ = self.model.predict(obs, deterministic=True)

        # Convert to battle order
        try:
            order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
            return order
        except Exception as e:
            print(f"[MultiDiscreteModelPlayer] Error converting action: {e}")
            return self.choose_random_doubles_move(battle)

    def _get_multidiscrete_mask(self, battle: DoubleBattle) -> np.ndarray:
        """Get action mask for MultiDiscrete action space."""
        from src.utils.action_masking import get_valid_actions_per_position
        mask0, mask1 = get_valid_actions_per_position(battle)
        # MaskablePPO with MultiDiscrete expects shape (n_envs, sum_of_action_dims)
        # For MultiDiscrete([107, 107]), it's shape (1, 214) flattened
        return np.concatenate([mask0, mask1])

    def _embed_battle(self, battle: DoubleBattle) -> np.ndarray:
        """Create observation embedding (must match training)."""
        # This should match what the MultiDiscrete model was trained with
        # Using the same embedding as DiscreteModelPlayer for now
        OBS_DIM = 64
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        idx = 0

        # Active Pokemon HP (2)
        for i in range(2):
            if i < len(battle.active_pokemon) and battle.active_pokemon[i] is not None:
                obs[idx] = battle.active_pokemon[i].current_hp_fraction
            idx += 1

        # Bench Pokemon HP (4)
        bench = [m for m in battle.team.values() if not m.active and not m.fainted]
        for i in range(4):
            if i < len(bench):
                obs[idx] = bench[i].current_hp_fraction
            idx += 1

        # Opponent active Pokemon HP (2)
        for i in range(2):
            if i < len(battle.opponent_active_pokemon) and battle.opponent_active_pokemon[i] is not None:
                obs[idx] = battle.opponent_active_pokemon[i].current_hp_fraction
            idx += 1

        # Opponent bench HP (4)
        opp_bench = [m for m in battle.opponent_team.values() if not m.active and not m.fainted]
        for i in range(4):
            if i < len(opp_bench):
                obs[idx] = opp_bench[i].current_hp_fraction
            idx += 1

        # Gimmick availability (8)
        for i in range(2):
            obs[idx] = float(battle.can_tera[i]) if i < len(battle.can_tera) else 0.0
            idx += 1
            obs[idx] = float(battle.can_mega_evolve[i]) if i < len(battle.can_mega_evolve) else 0.0
            idx += 1
            obs[idx] = float(battle.can_dynamax[i]) if i < len(battle.can_dynamax) else 0.0
            idx += 1
            obs[idx] = float(battle.can_z_move[i]) if i < len(battle.can_z_move) else 0.0
            idx += 1

        # Force switch (2)
        for i in range(2):
            obs[idx] = float(battle.force_switch[i]) if i < len(battle.force_switch) else 0.0
            idx += 1

        # Turn number (1)
        obs[idx] = min(battle.turn / 50.0, 1.0)

        return obs


async def run_battles(n_battles: int = 100):
    """Run battles between the two models."""
    print("=" * 60)
    print("Model vs Model Evaluation")
    print("=" * 60)
    print(f"\nDiscrete Model (MaskablePPO): {DISCRETE_MODEL_PATH.name}")
    print(f"MultiDiscrete Model (PPO): {MULTIDISCRETE_MODEL_PATH.name}")
    print(f"Number of battles: {n_battles}")
    print()

    # Check model files exist
    if not DISCRETE_MODEL_PATH.exists():
        print(f"Error: Discrete model not found at {DISCRETE_MODEL_PATH}")
        return
    if not MULTIDISCRETE_MODEL_PATH.exists():
        print(f"Error: MultiDiscrete model not found at {MULTIDISCRETE_MODEL_PATH}")
        return

    # Create players
    print("Loading models...")
    discrete_player = DiscreteModelPlayer(DISCRETE_MODEL_PATH)
    multidiscrete_player = MultiDiscreteModelPlayer(MULTIDISCRETE_MODEL_PATH)

    print(f"\nStarting {n_battles} battles...")
    print("-" * 60)

    # Run battles
    await discrete_player.battle_against(multidiscrete_player, n_battles=n_battles)

    # Results
    discrete_wins = discrete_player.n_won_battles
    multidiscrete_wins = multidiscrete_player.n_won_battles

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Discrete Model (Discrete 11449) wins:      {discrete_wins}/{n_battles} ({100*discrete_wins/n_battles:.1f}%)")
    print(f"MultiDiscrete Model (MultiDiscrete 107x107) wins: {multidiscrete_wins}/{n_battles} ({100*multidiscrete_wins/n_battles:.1f}%)")
    print("=" * 60)

    return {
        "discrete_wins": discrete_wins,
        "multidiscrete_wins": multidiscrete_wins,
        "total": n_battles,
    }


class RandomPlayer(Player):
    """Random baseline player for comparison."""

    def __init__(self, battle_format: str = "gen9vgc2026regf", team: Optional[str] = None, **kwargs):
        super().__init__(
            account_configuration=random_account("Random"),
            battle_format=battle_format,
            team=team or load_team(),
            accept_open_team_sheet=True,
            server_configuration=LocalhostServerConfiguration,
            **kwargs
        )

    def teampreview(self, battle: DoubleBattle) -> str:
        members = list(range(1, len(battle.team) + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, len(members))
        return "/team " + "".join(str(m) for m in members[:select_size])

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        return self.choose_random_doubles_move(battle)


async def run_vs_random(model_type: str, model_path: Path, n_battles: int = 100):
    """Run battles between a model and random player."""
    print("=" * 60)
    print(f"Model vs Random Evaluation")
    print("=" * 60)
    print(f"Model: {model_path.name}")
    print(f"Type: {model_type}")
    print(f"Number of battles: {n_battles}")
    print()

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model...")
    if model_type == "discrete":
        model_player = DiscreteModelPlayer(model_path)
    else:
        model_player = MultiDiscreteModelPlayer(model_path)

    random_player = RandomPlayer()

    print(f"\nStarting {n_battles} battles...")
    print("-" * 60)

    await model_player.battle_against(random_player, n_battles=n_battles)

    model_wins = model_player.n_won_battles
    random_wins = random_player.n_won_battles

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model wins:  {model_wins}/{n_battles} ({100*model_wins/n_battles:.1f}%)")
    print(f"Random wins: {random_wins}/{n_battles} ({100*random_wins/n_battles:.1f}%)")
    print("=" * 60)

    return {"model_wins": model_wins, "random_wins": random_wins, "total": n_battles}


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--battles", "-n", type=int, default=100, help="Number of battles to run")
    parser.add_argument("--discrete-model", type=str, default=None, help="Path to discrete model")
    parser.add_argument("--multidiscrete-model", type=str, default=None, help="Path to multidiscrete model")
    parser.add_argument("--vs-random", choices=["discrete", "multidiscrete"], default=None,
                        help="Run specified model against random instead of model vs model")
    args = parser.parse_args()

    # Override model paths if provided
    global DISCRETE_MODEL_PATH, MULTIDISCRETE_MODEL_PATH
    if args.discrete_model:
        DISCRETE_MODEL_PATH = Path(args.discrete_model)
    if args.multidiscrete_model:
        MULTIDISCRETE_MODEL_PATH = Path(args.multidiscrete_model)

    # Run evaluation
    if args.vs_random:
        model_path = DISCRETE_MODEL_PATH if args.vs_random == "discrete" else MULTIDISCRETE_MODEL_PATH
        asyncio.run(run_vs_random(args.vs_random, model_path, n_battles=args.battles))
    else:
        asyncio.run(run_battles(n_battles=args.battles))


if __name__ == "__main__":
    main()

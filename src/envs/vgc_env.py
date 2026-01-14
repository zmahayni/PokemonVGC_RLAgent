"""
VGC Environment for single-agent RL training with action masking.

This environment wraps poke-env to provide:
- Single-agent Gymnasium interface (opponent handled internally)
- Action masking for joint-legality constraints (required for MaskablePPO)
- Custom observation space for VGC doubles battles
- Reward shaping for RL training

Usage:
    from src.envs import VGCEnv
    from src.players import RandomVGCPlayer
    from sb3_contrib import MaskablePPO

    env = VGCEnv(opponent_type="random")
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
"""

import asyncio
import random
import sys
import threading
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union
from queue import Queue, Empty

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
from poke_env.concurrency import POKE_LOOP

from ..players.vgc_player import load_team, random_account, DEFAULT_TEAM_PATH
from ..utils.action_masking import get_action_mask, ACTION_SPACE_SIZE


# Observation dimension
OBS_DIM = 64


class _RLPlayer(Player):
    """
    Internal player class that bridges poke-env's async Player with gym's sync interface.

    This player puts battle states into a queue for the gym env to read,
    and reads actions from another queue to send back to the battle.
    """

    def __init__(
        self,
        battle_queue: Queue,
        action_queue: Queue,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._battle_queue = battle_queue
        self._action_queue = action_queue
        self._current_battle: Optional[DoubleBattle] = None

    def _battle_finished_callback(self, battle: DoubleBattle):
        """Called by poke-env when battle ends. Put final state in queue."""
        self._current_battle = battle
        self._battle_queue.put(battle)

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6)."""
        # Get team size (max 6 for VGC)
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        # VGC uses 4 Pokemon, ensure we don't exceed max_team_size
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> Awaitable[BattleOrder]:
        """
        Called by poke-env when it's time to choose a move.

        Puts the battle state in the queue and waits for an action.
        """
        return self._async_choose_move(battle)

    async def _async_choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Async version that bridges to the sync queue interface."""
        self._current_battle = battle

        # Put battle state in queue for gym env to read
        self._battle_queue.put(battle)

        # Wait for action from gym env
        while True:
            try:
                action = self._action_queue.get(timeout=0.1)
                break
            except Empty:
                await asyncio.sleep(0.01)

        # Convert action to order
        try:
            order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
        except Exception:
            # Fallback to random if action is invalid
            order = self.choose_random_doubles_move(battle)

        return order


class _RandomOpponent(Player):
    """Simple random opponent for training."""

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6)."""
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        return self.choose_random_doubles_move(battle)


class VGCEnv(Env):
    """
    Single-agent VGC Environment for RL training with action masking.

    This environment provides a Gymnasium interface for training RL agents
    on Pokemon VGC doubles battles. It handles:
    - Communication with Pokemon Showdown server via background thread
    - Action masking for invalid joint-action pairs
    - Observation embedding
    - Reward computation

    Note: Requires Pokemon Showdown server to be running:
        cd pokemon-showdown && node pokemon-showdown start --no-security
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        battle_format: str = "gen9vgc2026regf",
        team_path: Union[str, Path] = DEFAULT_TEAM_PATH,
        opponent_team_path: Optional[Union[str, Path]] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize VGC Environment.

        Args:
            battle_format: VGC format string
            team_path: Path to agent's team file
            opponent_team_path: Path to opponent's team file (uses same team if None)
            reward_config: Custom reward weights (optional)
        """
        super().__init__()

        self.battle_format = battle_format
        self.team_path = Path(team_path)
        self.opponent_team_path = Path(opponent_team_path) if opponent_team_path else self.team_path

        # Load teams
        self._team = load_team(self.team_path)
        self._opponent_team = load_team(self.opponent_team_path)

        # Communication queues
        self._battle_queue: Queue = Queue()
        self._action_queue: Queue = Queue()

        # Create players (will be initialized in reset)
        self._agent: Optional[_RLPlayer] = None
        self._opponent: Optional[_RandomOpponent] = None
        self._battle_task = None

        # Current state
        self._current_battle: Optional[DoubleBattle] = None
        self._episode_started = False

        # Action space: MultiDiscrete([107, 107]) for two active Pokemon
        self.action_space = MultiDiscrete([ACTION_SPACE_SIZE, ACTION_SPACE_SIZE])

        # Observation space: Box for continuous features
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # Reward configuration
        self._reward_config = reward_config or {
            "victory": 1.0,
            "defeat": -1.0,
            "fainted_value": 0.15,
            "hp_value": 0.05,
        }

        # Reward tracking
        self._last_reward_state: Optional[Dict[str, float]] = None

    @property
    def current_battle(self) -> Optional[DoubleBattle]:
        """Get the current battle state."""
        return self._current_battle

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """
        Get the action mask for the current state.

        For MaskablePPO with MultiDiscrete([107, 107]), this returns
        a concatenated mask of shape (214,) = (107 + 107,).

        Note: This provides per-position masks (independent), not joint masks.
        Joint-legality (e.g., both can't tera) is handled by the environment's
        fallback to random moves when an invalid pair is sampled.

        This method is required for sb3_contrib.MaskablePPO.
        """
        if self._current_battle is None:
            # Return all valid if no battle yet
            return np.ones(ACTION_SPACE_SIZE * 2, dtype=bool)

        try:
            from ..utils.action_masking import get_valid_actions_per_position
            mask0, mask1 = get_valid_actions_per_position(self._current_battle)
            # Concatenate masks for MultiDiscrete: [mask_pos0, mask_pos1]
            return np.concatenate([mask0, mask1])
        except Exception:
            # Fallback to all valid
            return np.ones(ACTION_SPACE_SIZE * 2, dtype=bool)

    def _ensure_players_exist(self):
        """Create players if they don't exist (lazy initialization)."""
        if self._agent is None:
            self._agent = _RLPlayer(
                battle_queue=self._battle_queue,
                action_queue=self._action_queue,
                account_configuration=random_account("Agent"),
                battle_format=self.battle_format,
                team=self._team,
                accept_open_team_sheet=True,
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
            )

        if self._opponent is None:
            self._opponent = _RandomOpponent(
                account_configuration=random_account("Opponent"),
                battle_format=self.battle_format,
                team=self._opponent_team,
                accept_open_team_sheet=True,
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        """
        Reset the environment and start a new battle.

        Args:
            seed: Random seed (optional)
            options: Additional options (optional)

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Clear any leftover items in queues from previous battle
        while not self._battle_queue.empty():
            try:
                self._battle_queue.get_nowait()
            except Empty:
                break
        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
            except Empty:
                break

        # Create players if needed (reuse across episodes)
        self._ensure_players_exist()

        # Reset battle state
        self._current_battle = None
        self._episode_started = False

        # Start new battle (reusing existing player connections)
        async def _run_battle():
            await self._agent.battle_against(self._opponent, n_battles=1)

        self._battle_task = asyncio.run_coroutine_threadsafe(_run_battle(), POKE_LOOP)

        # Wait for first battle state
        try:
            self._current_battle = self._battle_queue.get(timeout=30.0)
        except Empty:
            raise RuntimeError("Timeout waiting for battle to start. Is the server running?")

        self._episode_started = True
        self._last_reward_state = self._compute_reward_state()

        # Create observation
        obs = self._embed_battle(self._current_battle)

        info = {
            "battle_tag": self._current_battle.battle_tag if self._current_battle else None,
            "action_mask": self.action_masks(),
        }

        return obs, info

    def step(
        self,
        action: npt.NDArray[np.int64]
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action array of shape (2,) representing actions for both active Pokemon

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._episode_started or self._current_battle is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Send action to the player
        self._action_queue.put(np.array(action))

        # Wait for next battle state (should be fast now with _battle_finished_callback)
        terminated = False
        truncated = False

        try:
            self._current_battle = self._battle_queue.get(timeout=5.0)
        except Empty:
            # Check if battle finished
            if self._agent and self._agent._current_battle and self._agent._current_battle.finished:
                self._current_battle = self._agent._current_battle
                terminated = True
            else:
                truncated = True

        # Check if battle finished
        if self._current_battle and self._current_battle.finished:
            terminated = True

        # Compute reward
        reward = self._compute_reward()

        # Create observation
        obs = self._embed_battle(self._current_battle)

        info = {
            "battle_tag": self._current_battle.battle_tag if self._current_battle else None,
            "turn": self._current_battle.turn if self._current_battle else 0,
            "won": self._current_battle.won if self._current_battle else None,
        }

        if not terminated:
            info["action_mask"] = self.action_masks()

        return obs, reward, terminated, truncated, info

    def _embed_battle(self, battle: Optional[DoubleBattle]) -> npt.NDArray[np.float32]:
        """Create observation embedding from battle state."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if battle is None:
            return obs

        idx = 0

        # Active Pokemon HP (positions 0-1)
        for i in range(2):
            if i < len(battle.active_pokemon) and battle.active_pokemon[i] is not None:
                obs[idx] = battle.active_pokemon[i].current_hp_fraction
            idx += 1

        # Bench Pokemon HP (up to 4)
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

        # Gimmick availability per position (8 features)
        for i in range(2):
            obs[idx] = float(battle.can_tera[i]) if i < len(battle.can_tera) else 0.0
            idx += 1
            obs[idx] = float(battle.can_mega_evolve[i]) if i < len(battle.can_mega_evolve) else 0.0
            idx += 1
            obs[idx] = float(battle.can_dynamax[i]) if i < len(battle.can_dynamax) else 0.0
            idx += 1
            obs[idx] = float(battle.can_z_move[i]) if i < len(battle.can_z_move) else 0.0
            idx += 1

        # Force switch indicators (2)
        for i in range(2):
            obs[idx] = float(battle.force_switch[i]) if i < len(battle.force_switch) else 0.0
            idx += 1

        # Turn number normalized
        obs[idx] = min(battle.turn / 50.0, 1.0)
        idx += 1

        return obs

    def _compute_reward_state(self) -> Dict[str, float]:
        """Compute current state for differential reward calculation."""
        if self._current_battle is None:
            return {"our_hp": 0.0, "opp_hp": 0.0, "our_fainted": 0, "opp_fainted": 0}

        our_hp = sum(m.current_hp_fraction for m in self._current_battle.team.values())
        opp_hp = sum(m.current_hp_fraction for m in self._current_battle.opponent_team.values())
        our_fainted = sum(1 for m in self._current_battle.team.values() if m.fainted)
        opp_fainted = sum(1 for m in self._current_battle.opponent_team.values() if m.fainted)

        return {
            "our_hp": our_hp,
            "opp_hp": opp_hp,
            "our_fainted": our_fainted,
            "opp_fainted": opp_fainted,
        }

    def _compute_reward(self) -> float:
        """Compute reward for the current step."""
        if self._current_battle is None:
            return 0.0

        current_state = self._compute_reward_state()
        reward = 0.0

        # Win/loss bonus
        if self._current_battle.won:
            reward += self._reward_config["victory"]
        elif self._current_battle.lost:
            reward += self._reward_config["defeat"]
        elif self._last_reward_state is not None:
            # Differential rewards during battle
            our_hp_delta = current_state["our_hp"] - self._last_reward_state["our_hp"]
            opp_hp_delta = current_state["opp_hp"] - self._last_reward_state["opp_hp"]
            reward += (-opp_hp_delta + our_hp_delta) * self._reward_config["hp_value"]

            our_faint_delta = current_state["our_fainted"] - self._last_reward_state["our_fainted"]
            opp_faint_delta = current_state["opp_fainted"] - self._last_reward_state["opp_fainted"]
            reward += (opp_faint_delta - our_faint_delta) * self._reward_config["fainted_value"]

        self._last_reward_state = current_state
        return reward

    def render(self, mode: str = "human"):
        """Render the current battle state."""
        if self._current_battle is None:
            print("No active battle")
            return

        battle = self._current_battle
        print(f"\n--- Turn {battle.turn} ---")
        print(f"Active: {[m.species if m else 'None' for m in battle.active_pokemon]}")
        print(f"Opponent: {[m.species if m else 'None' for m in battle.opponent_active_pokemon]}")

    def close(self):
        """Clean up the environment and disconnect from server."""
        # Clear queues
        while not self._battle_queue.empty():
            try:
                self._battle_queue.get_nowait()
            except Empty:
                break
        while not self._action_queue.empty():
            try:
                self._action_queue.get_nowait()
            except Empty:
                break

        # Stop listening and disconnect players
        if self._agent is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._agent.ps_client.stop_listening(), POKE_LOOP
                ).result(timeout=5.0)
            except Exception:
                pass

        if self._opponent is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._opponent.ps_client.stop_listening(), POKE_LOOP
                ).result(timeout=5.0)
            except Exception:
                pass

        self._agent = None
        self._opponent = None
        self._current_battle = None
        self._episode_started = False

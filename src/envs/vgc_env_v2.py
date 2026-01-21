"""
VGC Environment V2 with improved observations and rewards.

Key improvements over VGCEnvDiscrete:
- Fixed bench Pokemon count (2 instead of 4 for VGC format)
- Rich observations: stat stages, status conditions, weather/terrain, type matchups
- Enhanced rewards: status infliction, stat boosts, speed control setup

Observation dimension: 86 (vs 64 in v1)
"""

import asyncio
import logging
import random
import sys
from pathlib import Path
from typing import Any, Awaitable, Dict, Optional, Tuple, Union
from queue import Queue, Empty

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.target import Target
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.ps_client import LocalhostServerConfiguration
from poke_env.concurrency import POKE_LOOP

from ..players.vgc_player import load_team, random_account, DEFAULT_TEAM_PATH
from ..utils.action_masking import (
    ACTION_SPACE_SIZE,
    get_action_mask_flat,
    flat_to_action_pair,
)
from ..utils.battle_features import (
    OBS_DIM_V2,
    extract_all_features,
    features_to_observation,
    STATUS_CONDITIONS,
)


class _RLPlayerV2(Player):
    """
    Internal player class that bridges poke-env's async Player with gym's sync interface.
    """

    _choose_move_count = 0

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
        """Called by poke-env when battle ends."""
        self._current_battle = battle
        self._battle_queue.put(battle)

    def teampreview(self, battle: DoubleBattle) -> str:
        """Random team preview selection (4 from 6)."""
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> Awaitable[BattleOrder]:
        return self._async_choose_move(battle)

    async def _async_choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Async version that bridges to the sync queue interface."""
        _RLPlayerV2._choose_move_count += 1
        call_id = _RLPlayerV2._choose_move_count

        self._current_battle = battle
        self._battle_queue.put(battle)

        # Wait for action with timeout tracking
        wait_iterations = 0
        while True:
            try:
                action = self._action_queue.get(timeout=0.1)
                break
            except Empty:
                wait_iterations += 1
                if wait_iterations % 100 == 0:  # Log every 10 seconds
                    print(f"[AGENT] choose_move #{call_id}: waiting for action ({wait_iterations * 0.1:.1f}s), "
                          f"turn={battle.turn}, tag={battle.battle_tag}", flush=True)
                if wait_iterations > 300:  # 30 second timeout
                    print(f"[AGENT] choose_move #{call_id}: TIMEOUT waiting for action, using random", flush=True)
                    return self.choose_random_doubles_move(battle)
                await asyncio.sleep(0.01)

        try:
            order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
        except Exception as e:
            print(f"[AGENT] choose_move #{call_id}: action_to_order failed ({e}), using random", flush=True)
            order = self.choose_random_doubles_move(battle)

        return order


class _RandomOpponentV2(Player):
    """Simple random opponent for training."""

    def teampreview(self, battle: DoubleBattle) -> str:
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        return self.choose_random_doubles_move(battle)


# Spread move targets that hit multiple opponents
_SPREAD_TARGETS = {
    Target.ALL_ADJACENT_FOES,
    Target.ALL_ADJACENT,
    Target.ALL,
}


class _HeuristicOpponentV2(Player):
    """
    Heuristic opponent that chooses moves based on damage estimation.

    Evaluates: base_power × type_effectiveness × STAB × stat_ratio × accuracy
    Falls back to random moves when heuristic can't find a good option.
    """

    _choose_move_count = 0

    def teampreview(self, battle: DoubleBattle) -> str:
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Choose moves based on damage estimation, fallback to random."""
        _HeuristicOpponentV2._choose_move_count += 1
        call_id = _HeuristicOpponentV2._choose_move_count

        try:
            order = self._choose_heuristic_move(battle)
            return order
        except Exception as e:
            print(f"[HEURISTIC] choose_move #{call_id}: exception ({e}), using random", flush=True)
            return self.choose_random_doubles_move(battle)

    def _choose_heuristic_move(self, battle: DoubleBattle) -> BattleOrder:
        """Internal heuristic move selection."""
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        orders = []
        switched_in = None
        used_heuristic = False  # Track if we successfully used heuristic

        for mon, moves, switches in zip(
            battle.active_pokemon,
            battle.available_moves,
            battle.available_switches
        ):
            available_switches = [s for s in switches if s != switched_in]

            if not mon or mon.fainted:
                orders.append(PassBattleOrder())
                continue

            if not moves:
                if available_switches:
                    switch_target = random.choice(available_switches)
                    orders.append(SingleBattleOrder(switch_target))
                    switched_in = switch_target
                else:
                    orders.append(PassBattleOrder())
                continue

            best_order = self._find_best_move(battle, mon, moves)

            if best_order is not None:
                orders.append(best_order)
                used_heuristic = True
            elif moves:
                # Fallback: pick first available move with a random valid target
                move = moves[0]
                targets = battle.get_possible_showdown_targets(move, mon)
                if targets:
                    orders.append(SingleBattleOrder(move, move_target=targets[0]))
                else:
                    orders.append(SingleBattleOrder(move))
            elif available_switches:
                switch_target = random.choice(available_switches)
                orders.append(SingleBattleOrder(switch_target))
                switched_in = switch_target
            else:
                orders.append(PassBattleOrder())

        # If we didn't use heuristic at all, fall back to random
        if not used_heuristic:
            return self.choose_random_doubles_move(battle)

        # Try to join orders
        if len(orders) >= 2 and (orders[0] or orders[1]):
            joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
            if joined:
                return joined[0]

        # Fallback to random if joining failed
        return self.choose_random_doubles_move(battle)

    def _find_best_move(self, battle: DoubleBattle, mon: Pokemon, moves) -> SingleBattleOrder:
        """Find the best move and target."""
        best_move = None
        best_target = None
        best_score = -1

        for move in moves:
            if move.base_power == 0:
                continue

            targets = battle.get_possible_showdown_targets(move, mon)

            for target in targets:
                if target not in [battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION]:
                    continue

                target_idx = target - 1
                if target_idx < 0 or target_idx >= len(battle.opponent_active_pokemon):
                    continue

                opp = battle.opponent_active_pokemon[target_idx]
                if not opp or opp.fainted:
                    continue

                score = self._estimate_damage(move, mon, opp)

                if move.target in _SPREAD_TARGETS:
                    other_opp = battle.opponent_active_pokemon[1 - target_idx]
                    if other_opp and not other_opp.fainted:
                        score *= 1.5

                if score > best_score:
                    best_score = score
                    best_move = move
                    best_target = target

        if best_move is not None and best_target is not None:
            return SingleBattleOrder(best_move, move_target=best_target)
        return None

    def _estimate_damage(self, move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """Estimate damage: base_power × type_mult × STAB × stat_ratio × accuracy"""
        base = move.base_power
        type_mult = defender.damage_multiplier(move)
        stab = 1.5 if move.type in attacker.types else 1.0

        if move.category == MoveCategory.PHYSICAL:
            atk_stat = attacker.base_stats.get("atk", 100)
            def_stat = defender.base_stats.get("def", 100)
        elif move.category == MoveCategory.SPECIAL:
            atk_stat = attacker.base_stats.get("spa", 100)
            def_stat = defender.base_stats.get("spd", 100)
        else:
            return 0

        stat_ratio = atk_stat / max(def_stat, 1)
        accuracy = move.accuracy if move.accuracy else 1.0

        return base * type_mult * stab * stat_ratio * accuracy


class _MaxPowerOpponentV2(Player):
    """Simple opponent that picks highest base power move (no type consideration)."""

    def teampreview(self, battle: DoubleBattle) -> str:
        team_size = min(len(battle.team), 6)
        members = list(range(1, team_size + 1))
        random.shuffle(members)
        select_size = min(battle.max_team_size or 4, 4, team_size)
        members = members[:select_size]
        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Pick highest base power move for each Pokemon."""
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        orders = []
        switched_in = None

        for mon, moves, switches in zip(
            battle.active_pokemon,
            battle.available_moves,
            battle.available_switches
        ):
            available_switches = [s for s in switches if s != switched_in]

            if not mon or mon.fainted:
                orders.append(PassBattleOrder())
                continue

            if not moves:
                if available_switches:
                    switch_target = random.choice(available_switches)
                    orders.append(SingleBattleOrder(switch_target))
                    switched_in = switch_target
                else:
                    orders.append(DefaultBattleOrder())
                continue

            # Find highest power move
            best_move = max(moves, key=lambda m: m.base_power, default=None)

            if best_move and best_move.base_power > 0:
                targets = battle.get_possible_showdown_targets(best_move, mon)
                opp_targets = [t for t in targets if t in [battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION]]
                target = random.choice(opp_targets) if opp_targets else (targets[0] if targets else None)
                if target:
                    orders.append(SingleBattleOrder(best_move, move_target=target))
                else:
                    orders.append(DefaultBattleOrder())
            else:
                orders.append(DefaultBattleOrder())

        if orders[0] or orders[1]:
            joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
            if joined:
                return joined[0]

        return self.choose_random_doubles_move(battle)


class VGCEnvV2(Env):
    """
    VGC Environment V2 with Discrete(11449) action space and improved observations.

    Key improvements:
    - Fixed bench count: 2 slots (VGC brings 4 Pokemon: 2 active + 2 bench)
    - Rich observations (86 features):
      * HP for all Pokemon (8)
      * Stat stages for active Pokemon (28)
      * Status conditions for active Pokemon (24)
      * Weather (4) and terrain (4) and trick room (1)
      * Type matchup hints (4)
      * Speed advantage (2)
      * Gimmick availability (8)
      * Force switch (2) and turn (1)
    - Enhanced rewards:
      * Status infliction bonus
      * Stat boost bonus
      * Speed control setup bonus (Tailwind/Trick Room)

    Requires Pokemon Showdown server:
        cd pokemon-showdown && node pokemon-showdown start --no-security
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        battle_format: str = "gen9vgc2026regf",
        team_path: Union[str, Path] = DEFAULT_TEAM_PATH,
        opponent_team_path: Optional[Union[str, Path]] = None,
        opponent_type: str = "random",
        reward_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize VGC Environment V2.

        Args:
            battle_format: VGC format string
            team_path: Path to agent's team file
            opponent_team_path: Path to opponent's team file (uses same team if None)
            opponent_type: Type of opponent - "random", "heuristic", or "max_power"
            reward_config: Custom reward weights (optional)
        """
        super().__init__()

        self.battle_format = battle_format
        self.team_path = Path(team_path)
        self.opponent_team_path = Path(opponent_team_path) if opponent_team_path else self.team_path
        self.opponent_type = opponent_type

        if opponent_type not in ["random", "heuristic", "max_power"]:
            raise ValueError(f"Unknown opponent_type: {opponent_type}. Use 'random', 'heuristic', or 'max_power'")

        self._team = load_team(self.team_path)
        self._opponent_team = load_team(self.opponent_team_path)

        self._battle_queue: Queue = Queue()
        self._action_queue: Queue = Queue()

        self._agent: Optional[_RLPlayerV2] = None
        self._opponent: Optional[Player] = None  # Can be random, heuristic, or max_power
        self._battle_task = None

        self._current_battle: Optional[DoubleBattle] = None
        self._episode_started = False
        self._episode_count = 0

        # Action space: Discrete(11449) = 107*107 for joint actions
        self.action_space = Discrete(ACTION_SPACE_SIZE * ACTION_SPACE_SIZE)

        # Observation space: 86 features (improved from 64)
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM_V2,),
            dtype=np.float32
        )

        # Reward configuration with new signals
        self._reward_config = reward_config or {
            "victory": 1.0,
            "defeat": -1.0,
            "fainted_value": 0.15,
            "hp_value": 0.05,
            # New reward signals
            "status_infliction": 0.02,  # Reward for inflicting status on opponent
            "stat_boost": 0.01,         # Reward for boosting our stats
            "speed_control": 0.03,      # Reward for Tailwind/Trick Room
        }

        # Reward tracking (expanded for new signals)
        self._last_reward_state: Optional[Dict[str, Any]] = None

    @property
    def current_battle(self) -> Optional[DoubleBattle]:
        return self._current_battle

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """Get joint-action mask for MaskablePPO. Shape: (11449,)"""
        if self._current_battle is None:
            return np.ones(ACTION_SPACE_SIZE * ACTION_SPACE_SIZE, dtype=bool)

        try:
            return get_action_mask_flat(self._current_battle)
        except Exception:
            return np.ones(ACTION_SPACE_SIZE * ACTION_SPACE_SIZE, dtype=bool)

    def _ensure_players_exist(self):
        """Lazy initialization of players."""
        if self._agent is None:
            self._agent = _RLPlayerV2(
                battle_queue=self._battle_queue,
                action_queue=self._action_queue,
                account_configuration=random_account("AgentV2"),
                battle_format=self.battle_format,
                team=self._team,
                accept_open_team_sheet=True,
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                log_level=logging.CRITICAL,
            )

        if self._opponent is None:
            # Select opponent class based on opponent_type
            opponent_classes = {
                "random": _RandomOpponentV2,
                "heuristic": _HeuristicOpponentV2,
                "max_power": _MaxPowerOpponentV2,
            }
            opponent_class = opponent_classes[self.opponent_type]

            self._opponent = opponent_class(
                account_configuration=random_account("OpponentV2"),
                battle_format=self.battle_format,
                team=self._opponent_team,
                accept_open_team_sheet=True,
                server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=1,
                log_level=logging.CRITICAL,
            )

    def _cleanup_players(self):
        """Clean up existing players (disconnect from server)."""
        if self._agent is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._agent.ps_client.stop_listening(), POKE_LOOP
                ).result(timeout=2.0)
            except Exception:
                pass
            self._agent = None

        if self._opponent is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._opponent.ps_client.stop_listening(), POKE_LOOP
                ).result(timeout=2.0)
            except Exception:
                pass
            self._opponent = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)

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

        self._current_battle = None
        self._episode_started = False
        self._episode_count += 1

        # Try to start battle, with retry on failure (recreate players if needed)
        max_retries = 3
        for attempt in range(max_retries):
            self._ensure_players_exist()

            async def _run_battle():
                await self._agent.battle_against(self._opponent, n_battles=1)

            self._battle_task = asyncio.run_coroutine_threadsafe(_run_battle(), POKE_LOOP)

            try:
                self._current_battle = self._battle_queue.get(timeout=30.0)
                break  # Success
            except Empty:
                print(f"[ENV] Episode #{self._episode_count}: attempt {attempt+1}/{max_retries} "
                      f"TIMEOUT waiting for battle to start", flush=True)
                if attempt < max_retries - 1:
                    # Recreate players for next attempt
                    print("[ENV] Recreating players...", flush=True)
                    self._cleanup_players()
                else:
                    print("[ENV] All retries failed!", flush=True)
                    print(f"[ENV] Agent choose_move calls: {_RLPlayerV2._choose_move_count}", flush=True)
                    print(f"[ENV] Heuristic choose_move calls: {_HeuristicOpponentV2._choose_move_count}", flush=True)
                    raise RuntimeError("Timeout waiting for battle to start after retries. Is the server running?")

        self._episode_started = True
        self._last_reward_state = self._compute_reward_state()

        obs = self._embed_battle(self._current_battle)
        info = {
            "battle_tag": self._current_battle.battle_tag if self._current_battle else None,
            "action_mask": self.action_masks(),
        }

        return obs, info

    _step_count = 0

    def step(
        self,
        action: int
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if not self._episode_started or self._current_battle is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        VGCEnvV2._step_count += 1
        step_id = VGCEnvV2._step_count

        # Convert flat action to action pair
        action_pair = flat_to_action_pair(action)
        self._action_queue.put(np.array(action_pair))

        terminated = False
        truncated = False

        try:
            self._current_battle = self._battle_queue.get(timeout=5.0)
        except Empty:
            print(f"[ENV] step #{step_id}: timeout waiting for battle state, "
                  f"turn={self._current_battle.turn if self._current_battle else '?'}, "
                  f"tag={self._current_battle.battle_tag if self._current_battle else '?'}", flush=True)
            if self._agent and self._agent._current_battle and self._agent._current_battle.finished:
                self._current_battle = self._agent._current_battle
                terminated = True
            else:
                truncated = True

        if self._current_battle and self._current_battle.finished:
            terminated = True

        reward = self._compute_reward()
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
        """Create observation using battle_features extraction."""
        if battle is None:
            return np.zeros(OBS_DIM_V2, dtype=np.float32)

        try:
            features = extract_all_features(battle)
            return features_to_observation(features)
        except Exception:
            # Fallback to zeros if extraction fails
            return np.zeros(OBS_DIM_V2, dtype=np.float32)

    def _compute_reward_state(self) -> Dict[str, Any]:
        """Compute current state for differential reward calculation (expanded)."""
        if self._current_battle is None:
            return {
                "our_hp": 0.0,
                "opp_hp": 0.0,
                "our_fainted": 0,
                "opp_fainted": 0,
                "opp_status_count": 0,
                "our_boost_sum": 0,
                "trick_room_active": False,
                "tailwind_active": False,
            }

        battle = self._current_battle

        our_hp = sum(m.current_hp_fraction for m in battle.team.values())
        opp_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())
        our_fainted = sum(1 for m in battle.team.values() if m.fainted)
        opp_fainted = sum(1 for m in battle.opponent_team.values() if m.fainted)

        # Count opponent status conditions
        opp_status_count = 0
        for mon in battle.opponent_team.values():
            if mon.status is not None and mon.status in STATUS_CONDITIONS:
                opp_status_count += 1

        # Sum our stat boosts (positive boosts only)
        our_boost_sum = 0
        for mon in battle.active_pokemon:
            if mon is not None:
                for stat, val in mon.boosts.items():
                    if val > 0:
                        our_boost_sum += val

        # Check speed control
        trick_room_active = Field.TRICK_ROOM in battle.fields
        # Tailwind is a side condition, check if we have it
        tailwind_active = False
        if hasattr(battle, 'side_conditions'):
            # Side conditions use SideCondition enum
            from poke_env.battle.side_condition import SideCondition
            tailwind_active = SideCondition.TAILWIND in battle.side_conditions

        return {
            "our_hp": our_hp,
            "opp_hp": opp_hp,
            "our_fainted": our_fainted,
            "opp_fainted": opp_fainted,
            "opp_status_count": opp_status_count,
            "our_boost_sum": our_boost_sum,
            "trick_room_active": trick_room_active,
            "tailwind_active": tailwind_active,
        }

    def _compute_reward(self) -> float:
        """Compute reward with enhanced signals."""
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

            # HP changes
            our_hp_delta = current_state["our_hp"] - self._last_reward_state["our_hp"]
            opp_hp_delta = current_state["opp_hp"] - self._last_reward_state["opp_hp"]
            reward += (-opp_hp_delta + our_hp_delta) * self._reward_config["hp_value"]

            # Faint changes
            our_faint_delta = current_state["our_fainted"] - self._last_reward_state["our_fainted"]
            opp_faint_delta = current_state["opp_fainted"] - self._last_reward_state["opp_fainted"]
            reward += (opp_faint_delta - our_faint_delta) * self._reward_config["fainted_value"]

            # NEW: Status infliction reward
            status_delta = current_state["opp_status_count"] - self._last_reward_state["opp_status_count"]
            if status_delta > 0:
                reward += status_delta * self._reward_config["status_infliction"]

            # NEW: Stat boost reward
            boost_delta = current_state["our_boost_sum"] - self._last_reward_state["our_boost_sum"]
            if boost_delta > 0:
                reward += boost_delta * self._reward_config["stat_boost"]

            # NEW: Speed control setup reward
            if current_state["trick_room_active"] and not self._last_reward_state["trick_room_active"]:
                reward += self._reward_config["speed_control"]
            if current_state["tailwind_active"] and not self._last_reward_state["tailwind_active"]:
                reward += self._reward_config["speed_control"]

        self._last_reward_state = current_state
        return reward

    def render(self, mode: str = "human"):
        if self._current_battle is None:
            print("No active battle")
            return

        battle = self._current_battle
        print(f"\n--- Turn {battle.turn} ---")
        print(f"Active: {[m.species if m else 'None' for m in battle.active_pokemon]}")
        print(f"Opponent: {[m.species if m else 'None' for m in battle.opponent_active_pokemon]}")

    def close(self):
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

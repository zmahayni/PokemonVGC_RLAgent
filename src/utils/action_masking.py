"""
Action masking utilities for VGC doubles battles.

This module provides functions to compute valid action masks for the
MultiDiscrete([107, 107]) action space used by DoublesEnv.

Key concepts:
- Gen9 has 107 possible actions per position (pass + 6 switches + 4*5*5 moves with targets/gimmicks)
- Not all action pairs are valid due to joint-legality constraints
- We can either mask per-position (faster) or mask the full joint space (more accurate)
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder


# Action space constants for Gen 9
NUM_SWITCHES = 6
NUM_MOVES = 4
NUM_TARGETS = 5
NUM_GIMMICKS = 5  # none, mega, z-move, dynamax, tera
ACTION_SPACE_SIZE = 1 + NUM_SWITCHES + NUM_MOVES * NUM_TARGETS * NUM_GIMMICKS  # = 107


def decode_action(action: int) -> str:
    """
    Decode a single position's action into human-readable form.

    Args:
        action: Action index (-2 to 106)

    Returns:
        Human-readable description of the action
    """
    if action == -2:
        return "default"
    if action == -1:
        return "forfeit"
    if action == 0:
        return "pass"
    if 1 <= action <= 6:
        return f"switch to mon {action}"

    # Move action
    action_offset = action - 7
    gimmick = action_offset // 20
    remainder = action_offset % 20
    move_idx = remainder // 5
    target_offset = remainder % 5
    target = target_offset - 2

    gimmick_names = ["", "+mega", "+zmove", "+dynamax", "+tera"]
    gimmick_str = gimmick_names[gimmick] if gimmick < len(gimmick_names) else f"+gimmick{gimmick}"

    target_names = {-2: "ally0", -1: "ally1/self", 0: "no_target", 1: "opp0", 2: "opp1"}
    target_str = target_names.get(target, f"target{target}")

    return f"move{move_idx+1} -> {target_str}{gimmick_str}"


def get_valid_actions_per_position(
    battle: DoubleBattle
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Get valid action masks for each position independently.

    This is a fast approximation - it doesn't account for joint-legality
    constraints between positions. Use get_action_mask() for full accuracy.

    Args:
        battle: Current battle state

    Returns:
        Tuple of (mask_pos0, mask_pos1) where each is a boolean array of shape (107,)
    """
    mask0 = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask1 = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

    # Get valid orders for each position
    valid_orders_0 = battle.valid_orders[0]
    valid_orders_1 = battle.valid_orders[1]

    # Convert orders to actions for position 0
    for order in valid_orders_0:
        try:
            # Create a dummy DoubleBattleOrder to get the action
            from poke_env.player.battle_order import PassBattleOrder
            temp_order = DoubleBattleOrder(order, PassBattleOrder())
            action = DoublesEnv.order_to_action(temp_order, battle, fake=True)
            if 0 <= action[0] < ACTION_SPACE_SIZE:
                mask0[action[0]] = True
        except (ValueError, IndexError):
            pass

    # Convert orders to actions for position 1
    for order in valid_orders_1:
        try:
            from poke_env.player.battle_order import PassBattleOrder
            temp_order = DoubleBattleOrder(PassBattleOrder(), order)
            action = DoublesEnv.order_to_action(temp_order, battle, fake=True)
            if 0 <= action[1] < ACTION_SPACE_SIZE:
                mask1[action[1]] = True
        except (ValueError, IndexError):
            pass

    return mask0, mask1


def get_action_mask(battle: DoubleBattle) -> npt.NDArray[np.bool_]:
    """
    Get the full joint-action mask for valid action pairs.

    This properly accounts for joint-legality constraints (e.g., both
    Pokemon can't use tera in the same turn, can't switch to the same mon).

    Args:
        battle: Current battle state

    Returns:
        Boolean array of shape (107, 107) where mask[a0, a1] is True if
        the action pair (a0, a1) is valid.
    """
    mask = np.zeros((ACTION_SPACE_SIZE, ACTION_SPACE_SIZE), dtype=bool)

    # Get all joint-legal orders
    valid_orders_0 = battle.valid_orders[0]
    valid_orders_1 = battle.valid_orders[1]
    joint_legal = DoubleBattleOrder.join_orders(valid_orders_0, valid_orders_1)

    # Convert each joint-legal order to action pair
    for double_order in joint_legal:
        try:
            action = DoublesEnv.order_to_action(double_order, battle, fake=True)
            a0, a1 = action[0], action[1]
            if 0 <= a0 < ACTION_SPACE_SIZE and 0 <= a1 < ACTION_SPACE_SIZE:
                mask[a0, a1] = True
        except (ValueError, IndexError):
            pass

    return mask


def get_action_mask_flat(battle: DoubleBattle) -> npt.NDArray[np.bool_]:
    """
    Get the joint-action mask as a flattened 1D array.

    This is useful for environments that use a single Discrete action space
    instead of MultiDiscrete. The action index maps to (a0, a1) as:
        a0 = action_idx // 107
        a1 = action_idx % 107

    Args:
        battle: Current battle state

    Returns:
        Boolean array of shape (107*107,) = (11449,)
    """
    mask_2d = get_action_mask(battle)
    return mask_2d.flatten()


def action_pair_to_flat(a0: int, a1: int) -> int:
    """Convert action pair to flat index."""
    return a0 * ACTION_SPACE_SIZE + a1


def flat_to_action_pair(flat_idx: int) -> Tuple[int, int]:
    """Convert flat index to action pair."""
    a0 = flat_idx // ACTION_SPACE_SIZE
    a1 = flat_idx % ACTION_SPACE_SIZE
    return a0, a1


def get_num_valid_actions(battle: DoubleBattle) -> int:
    """Get the number of valid joint actions in the current state."""
    valid_orders_0 = battle.valid_orders[0]
    valid_orders_1 = battle.valid_orders[1]
    joint_legal = DoubleBattleOrder.join_orders(valid_orders_0, valid_orders_1)
    return len(joint_legal)

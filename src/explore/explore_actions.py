"""
Explore the action space for DoublesEnv from poke-env.

This script documents how actions map to orders and explores joint-legality constraints.

Action encoding for doubles (per position):
    -2: default order
    -1: forfeit
    0: pass
    1-6: switch to team member 1-6
    7+: moves with targets and gimmicks

Move action formula: action = 7 + (move_idx * 5) + target_offset + (gimmick * 20)
    - move_idx: 0-3 (4 moves)
    - target_offset: 0-4 (corresponding to targets -2, -1, 0, 1, 2)
    - gimmick: 0=none, 1=mega, 2=z-move, 3=dynamax, 4=tera

Target positions:
    -2: ally at position 0 (first ally)
    -1: ally at position 1 (second ally / self)
    0: no specific target (spread moves, self-targeting)
    1: opponent at position 0
    2: opponent at position 1

Run the Showdown server first:
    cd pokemon-showdown && node pokemon-showdown start --no-security

Then run this script:
    python src/explore/explore_actions.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, 'poke-env/src')

import numpy as np
from poke_env.player import RandomPlayer, Player
from poke_env.ps_client import LocalhostServerConfiguration
from poke_env.battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder

# Load team
TEAM_PATH = Path(__file__).parent.parent.parent / "teams" / "team.txt"

def load_team() -> str:
    if TEAM_PATH.exists():
        return TEAM_PATH.read_text()
    return None


def decode_action(action: int) -> str:
    """Decode a single position's action into human-readable form."""
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


def print_action_space_reference():
    """Print a reference guide for the action space."""
    print("\n" + "=" * 80)
    print("ACTION SPACE REFERENCE (Gen 9, per position)")
    print("=" * 80)

    print("\nTotal action size: 107 (1 + 6 + 4*5*5)")
    print("\nSpecial actions:")
    print("  -2: default (let game choose)")
    print("  -1: forfeit")
    print("   0: pass")

    print("\nSwitch actions:")
    for i in range(1, 7):
        print(f"   {i}: switch to team member {i}")

    print("\nMove actions (7-106):")
    print("  Formula: 7 + (move_idx * 5) + target_offset + (gimmick * 20)")
    print("  move_idx: 0-3, target_offset: 0-4 (targets -2 to 2), gimmick: 0-4")

    print("\n  Without gimmick (actions 7-26):")
    for move in range(4):
        start = 7 + move * 5
        print(f"    Move {move+1}: actions {start}-{start+4}")
        for target in range(5):
            action = start + target
            target_val = target - 2
            print(f"      {action}: move{move+1} -> target {target_val}")

    print("\n  With Tera (actions 87-106):")
    for move in range(4):
        start = 87 + move * 5
        print(f"    Move {move+1}+tera: actions {start}-{start+4}")


class ActionExplorer(Player):
    """A player that explores action mappings and joint-legality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.turn_count = 0

    def choose_move(self, battle: DoubleBattle):
        """Explore actions then choose randomly."""
        self.turn_count += 1

        if self.turn_count <= 3:  # Only explore first 3 turns
            self._explore_actions(battle)

        return self.choose_random_doubles_move(battle)

    def _explore_actions(self, battle: DoubleBattle):
        """Explore and document action space for current battle state."""
        print("\n" + "=" * 80)
        print(f"ACTION EXPLORATION - Turn {battle.turn}")
        print("=" * 80)

        # Get valid orders per position
        valid_orders_0 = battle.valid_orders[0]
        valid_orders_1 = battle.valid_orders[1]

        print(f"\nValid orders for position 0: {len(valid_orders_0)}")
        print(f"Valid orders for position 1: {len(valid_orders_1)}")

        # Convert orders to actions and show mapping
        print("\n--- Position 0 orders -> actions ---")
        for order in valid_orders_0[:5]:  # First 5
            try:
                # Create a DoubleBattleOrder with this order and pass for position 1
                from poke_env.player.battle_order import PassBattleOrder
                temp_order = DoubleBattleOrder(order, PassBattleOrder())
                action = DoublesEnv.order_to_action(temp_order, battle, fake=True)
                print(f"  {str(order):40} -> action {action[0]:3} ({decode_action(action[0])})")
            except Exception as e:
                print(f"  {str(order):40} -> ERROR: {e}")

        if len(valid_orders_0) > 5:
            print(f"  ... and {len(valid_orders_0) - 5} more")

        print("\n--- Position 1 orders -> actions ---")
        for order in valid_orders_1[:5]:  # First 5
            try:
                from poke_env.player.battle_order import PassBattleOrder
                temp_order = DoubleBattleOrder(PassBattleOrder(), order)
                action = DoublesEnv.order_to_action(temp_order, battle, fake=True)
                print(f"  {str(order):40} -> action {action[1]:3} ({decode_action(action[1])})")
            except Exception as e:
                print(f"  {str(order):40} -> ERROR: {e}")

        if len(valid_orders_1) > 5:
            print(f"  ... and {len(valid_orders_1) - 5} more")

        # Explore joint-legality
        print("\n--- JOINT-LEGALITY EXPLORATION ---")
        joint_legal = DoubleBattleOrder.join_orders(valid_orders_0, valid_orders_1)
        print(f"Total joint-legal combinations: {len(joint_legal)}")
        print(f"(vs {len(valid_orders_0)} x {len(valid_orders_1)} = {len(valid_orders_0) * len(valid_orders_1)} if all pairs were legal)")

        # Show some joint-legal examples
        print("\nSample joint-legal orders:")
        for double_order in joint_legal[:5]:
            try:
                action = DoublesEnv.order_to_action(double_order, battle, fake=True)
                print(f"  [{action[0]:3}, {action[1]:3}]: {double_order.message}")
            except Exception as e:
                print(f"  ERROR: {e}")

        # Find incompatible pairs (if any exist)
        print("\n--- INCOMPATIBILITY CHECK ---")
        incompatible_count = 0
        incompatible_examples = []

        for o0 in valid_orders_0[:10]:  # Sample
            for o1 in valid_orders_1[:10]:
                joined = DoubleBattleOrder.join_orders([o0], [o1])
                if not joined:
                    incompatible_count += 1
                    if len(incompatible_examples) < 3:
                        incompatible_examples.append((o0, o1))

        if incompatible_examples:
            print(f"Found {incompatible_count} incompatible pairs in sample (out of 100)")
            print("Examples of incompatible pairs:")
            for o0, o1 in incompatible_examples:
                print(f"  [{str(o0):35}] + [{str(o1):35}] = INVALID")
                # Try to identify why
                reasons = []
                if hasattr(o0, 'terastallize') and o0.terastallize and hasattr(o1, 'terastallize') and o1.terastallize:
                    reasons.append("both tera")
                if hasattr(o0, 'mega') and o0.mega and hasattr(o1, 'mega') and o1.mega:
                    reasons.append("both mega")
                if hasattr(o0, 'order') and hasattr(o1, 'order'):
                    from poke_env.battle.pokemon import Pokemon
                    if isinstance(o0.order, Pokemon) and isinstance(o1.order, Pokemon):
                        if o0.order.species == o1.order.species:
                            reasons.append("both switch to same mon")
                if reasons:
                    print(f"    Reason: {', '.join(reasons)}")
        else:
            print("No incompatible pairs found in sample - all combinations may be legal this turn")


async def run_action_exploration(battle_format: str = "gen9vgc2026regf", n_battles: int = 1):
    """Run battles with action exploration."""
    print_action_space_reference()

    print("\n\n" + "=" * 80)
    print("LIVE ACTION EXPLORATION")
    print("=" * 80)
    print(f"\nFormat: {battle_format}")

    team = load_team()
    if team:
        print(f"Using team from: {TEAM_PATH}")
    print("Running battle to explore actions in real game states...\n")

    explorer = ActionExplorer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
    )

    await explorer.battle_against(opponent, n_battles=n_battles)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

    await explorer.ps_client.stop_listening()
    await opponent.ps_client.stop_listening()


async def main():
    await run_action_exploration(
        battle_format="gen9vgc2026regf",
        n_battles=1
    )


if __name__ == "__main__":
    asyncio.run(main())

"""
Explore the DoubleBattle state object from poke-env.

This script creates a player that prints detailed battle state at each turn,
helping us understand what information is available for RL observations.

Run the Showdown server first:
    cd pokemon-showdown && node pokemon-showdown start --no-security

Then run this script:
    python src/explore/explore_battle_state.py
"""

import asyncio
import sys
import random
from pathlib import Path

sys.path.insert(0, 'poke-env/src')

from poke_env.player import RandomPlayer, Player
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from poke_env.battle import DoubleBattle

# Load team
TEAM_PATH = Path(__file__).parent.parent.parent / "teams" / "team.txt"

def random_name(prefix: str) -> AccountConfiguration:
    """Generate unique player name to avoid conflicts."""
    return AccountConfiguration(f"{prefix}{random.randint(1000, 9999)}", None)

def load_team() -> str:
    if TEAM_PATH.exists():
        return TEAM_PATH.read_text()
    return None


class VGCRandomPlayer(RandomPlayer):
    """RandomPlayer that properly handles VGC team preview."""

    def teampreview(self, battle) -> str:
        members = list(range(1, len(battle.team) + 1))
        random.shuffle(members)
        if battle.max_team_size:
            members = members[:battle.max_team_size]
        return "/team " + "".join([str(c) for c in members])


class BattleStateExplorer(Player):
    """A player that prints detailed battle state each turn."""

    def __init__(self, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def teampreview(self, battle: DoubleBattle) -> str:
        """Override to properly handle VGC team preview (select 4 from 6)."""
        members = list(range(1, len(battle.team) + 1))
        random.shuffle(members)
        # Limit to max_team_size (4 for VGC)
        if battle.max_team_size:
            members = members[:battle.max_team_size]
        return "/team " + "".join([str(c) for c in members])

    def choose_move(self, battle: DoubleBattle):
        """Print battle state then choose randomly."""
        if self.verbose:
            self._print_battle_state(battle)

        # Use parent's random choice
        return self.choose_random_doubles_move(battle)

    def _print_battle_state(self, battle: DoubleBattle):
        """Print comprehensive battle state information."""
        print("\n" + "=" * 80)
        print(f"TURN {battle.turn}")
        print("=" * 80)

        # Basic battle info
        print(f"\nBattle tag: {battle.battle_tag}")
        print(f"Format: {battle.format}")
        print(f"Finished: {battle.finished}, Won: {battle.won}, Lost: {battle.lost}")

        # Active Pokemon
        print("\n--- ACTIVE POKEMON ---")
        self._print_active_pokemon("My Pokemon", battle.active_pokemon)
        self._print_active_pokemon("Opponent Pokemon", battle.opponent_active_pokemon)

        # Available actions per position
        print("\n--- AVAILABLE ACTIONS ---")
        for i, (moves, switches) in enumerate(zip(battle.available_moves, battle.available_switches)):
            print(f"\nPosition {i} (slot {'a' if i == 0 else 'b'}):")
            print(f"  Force switch: {battle.force_switch[i]}")
            print(f"  Trapped: {battle.trapped[i]}")
            print(f"  Can Tera: {battle.can_tera[i]}")
            print(f"  Available moves ({len(moves)}):")
            for move in moves:
                targets = self._get_move_targets(move, battle)
                print(f"    - {move.id}: power={move.base_power}, type={move.type.name}, "
                      f"acc={move.accuracy}, category={move.category.name}, targets={targets}")
            print(f"  Available switches ({len(switches)}):")
            for mon in switches:
                print(f"    - {mon.species}: HP={mon.current_hp_fraction:.1%}")

        # Valid orders (joint-legal action combinations)
        print("\n--- VALID ORDERS (joint-legal) ---")
        print(f"Position 0 orders: {len(battle.valid_orders[0])}")
        print(f"Position 1 orders: {len(battle.valid_orders[1])}")
        # Print first few examples
        for i, orders in enumerate(battle.valid_orders):
            print(f"\nPosition {i} sample orders:")
            for order in orders[:3]:  # First 3
                print(f"    {order.message}")
            if len(orders) > 3:
                print(f"    ... and {len(orders) - 3} more")

        # Team information
        print("\n--- MY TEAM ---")
        for name, mon in battle.team.items():
            status = f", status={mon.status.name}" if mon.status else ""
            tera = f", tera={mon.tera_type.name}" if mon.is_terastallized else ""
            print(f"  {mon.species}: HP={mon.current_hp_fraction:.1%}, "
                  f"fainted={mon.fainted}, active={mon.active}{status}{tera}")

        # Opponent team (what we've seen)
        print("\n--- OPPONENT TEAM (revealed) ---")
        for name, mon in battle.opponent_team.items():
            status = f", status={mon.status.name}" if mon.status else ""
            print(f"  {mon.species}: HP={mon.current_hp_fraction:.1%}, "
                  f"fainted={mon.fainted}, active={mon.active}{status}")

        # Field conditions
        print("\n--- FIELD CONDITIONS ---")
        print(f"  Weather: {dict(battle.weather) if battle.weather else 'None'}")
        print(f"  Fields (terrain): {dict(battle.fields) if battle.fields else 'None'}")
        print(f"  My side conditions: {dict(battle.side_conditions) if battle.side_conditions else 'None'}")
        print(f"  Opp side conditions: {dict(battle.opponent_side_conditions) if battle.opponent_side_conditions else 'None'}")

    def _print_active_pokemon(self, label: str, pokemon_list):
        """Print info about active Pokemon."""
        print(f"\n{label}:")
        for i, mon in enumerate(pokemon_list):
            if mon is None:
                print(f"  Position {i}: Empty (fainted or not sent out)")
            else:
                types = "/".join(t.name for t in mon.types if t)
                status = f", status={mon.status.name}" if mon.status else ""
                boosts = {k: v for k, v in mon.boosts.items() if v != 0}
                boost_str = f", boosts={boosts}" if boosts else ""
                print(f"  Position {i}: {mon.species}")
                print(f"    HP: {mon.current_hp_fraction:.1%}, Types: {types}{status}{boost_str}")
                print(f"    Ability: {mon.ability}, Item: {mon.item}")

    def _get_move_targets(self, move, battle):
        """Get possible targets for a move."""
        try:
            # Use poke-env's targeting logic
            targets = move.target.name if move.target else "unknown"
            return targets
        except:
            return "unknown"


async def run_exploration(battle_format: str = "gen9vgc2026regf", n_battles: int = 1):
    """Run battles with state exploration."""
    print(f"\nStarting battle exploration with format: {battle_format}")
    print("This will print detailed state information for each turn.\n")

    team = load_team()
    if team:
        print(f"Using team from: {TEAM_PATH}\n")

    explorer = BattleStateExplorer(
        account_configuration=random_name("Explorer"),
        battle_format=battle_format,
        verbose=True,
        team=team,
        accept_open_team_sheet=True,  # Required for VGC formats
    )
    opponent = VGCRandomPlayer(
        account_configuration=random_name("Opponent"),
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,  # Required for VGC formats
    )

    await explorer.battle_against(opponent, n_battles=n_battles)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)
    print(f"Explorer wins: {explorer.n_won_battles}/{explorer.n_finished_battles}")

    await explorer.ps_client.stop_listening()
    await opponent.ps_client.stop_listening()


async def main():
    # Using VGC 2026 Reg F with your team
    await run_exploration(
        battle_format="gen9vgc2026regf",
        n_battles=1
    )


if __name__ == "__main__":
    asyncio.run(main())

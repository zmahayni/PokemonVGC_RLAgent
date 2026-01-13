"""
Test connection to Pokemon Showdown server with poke-env.

Run the Showdown server first:
    cd pokemon-showdown && node pokemon-showdown start --no-security

Then run this script:
    python src/explore/test_connection.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, 'poke-env/src')

from poke_env.player import RandomPlayer
from poke_env.ps_client import LocalhostServerConfiguration


# Try different format strings to find what works
# VGC formats typically follow pattern: gen9vgc2026regf, gen9vgc2024regg, etc.
FORMATS_TO_TRY = [
    ("gen9vgc2026regf", True),       # User's target - Reg F (needs team)
    ("gen9randomdoublesbattle", False),  # Random doubles (no team needed, for basic connectivity test)
]

# Path to team file
TEAM_PATH = Path(__file__).parent.parent.parent / "teams" / "team.txt"


def load_team() -> str:
    """Load team from file."""
    if TEAM_PATH.exists():
        return TEAM_PATH.read_text()
    return None


async def test_format(battle_format: str, needs_team: bool, n_battles: int = 3) -> bool:
    """Test if a format works by running random battles."""
    print(f"\n{'='*60}")
    print(f"Testing format: {battle_format}")
    print(f"{'='*60}")

    team = load_team() if needs_team else None
    if needs_team and team is None:
        print(f"SKIPPED: Format requires team but teams/team.txt not found")
        return False

    if needs_team:
        print(f"Using team from: {TEAM_PATH}")

    try:
        p1 = RandomPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
            team=team,
        )
        p2 = RandomPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
            team=team,
        )

        print(f"Players created, starting {n_battles} battles...")
        await p1.battle_against(p2, n_battles=n_battles)

        print(f"SUCCESS!")
        print(f"  P1 wins: {p1.n_won_battles}/{p1.n_finished_battles}")
        print(f"  P2 wins: {p2.n_won_battles}/{p2.n_finished_battles}")

        # Clean up
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()

        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


async def main():
    print("=" * 60)
    print("Pokemon Showdown Connection Test")
    print("=" * 60)
    print("\nMake sure the Showdown server is running:")
    print("  cd pokemon-showdown && node pokemon-showdown start --no-security")
    print()

    working_formats = []

    for fmt, needs_team in FORMATS_TO_TRY:
        try:
            success = await asyncio.wait_for(
                test_format(fmt, needs_team, n_battles=2),
                timeout=30
            )
            if success:
                working_formats.append(fmt)
        except asyncio.TimeoutError:
            print(f"TIMEOUT for format: {fmt}")
        except Exception as e:
            print(f"Error testing {fmt}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if working_formats:
        print("Working formats:")
        for fmt in working_formats:
            print(f"  - {fmt}")
    else:
        print("No formats worked! Check if Showdown server is running.")


if __name__ == "__main__":
    asyncio.run(main())

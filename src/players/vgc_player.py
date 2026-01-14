"""
VGC Player base class for Pokemon VGC doubles battles.

This module provides a properly configured Player class for VGC formats,
handling team preview, open team sheets, and other VGC-specific requirements.
"""

import random
import sys
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle import DoubleBattle
from poke_env.teambuilder import Teambuilder


# Default team path
DEFAULT_TEAM_PATH = Path(__file__).parent.parent.parent / "teams" / "team.txt"


def load_team(team_path: Union[str, Path] = DEFAULT_TEAM_PATH) -> str:
    """Load team from file."""
    path = Path(team_path)
    if not path.exists():
        raise FileNotFoundError(f"Team file not found: {path}")
    return path.read_text()


def random_account(prefix: str = "VGC") -> AccountConfiguration:
    """Generate a random account configuration to avoid name collisions."""
    return AccountConfiguration(f"{prefix}{random.randint(1000, 9999)}", None)


class VGCPlayer(Player):
    """
    Base player class for VGC doubles battles.

    This class properly configures all VGC-specific requirements:
    - accept_open_team_sheet=True (required for VGC)
    - Team loading from file
    - Random team preview selection (4 from 6)

    Subclass this and implement choose_move() for your RL agent.

    Example:
        class MyAgent(VGCPlayer):
            def choose_move(self, battle: DoubleBattle):
                # Your RL logic here
                return self.choose_random_doubles_move(battle)
    """

    def __init__(
        self,
        team_path: Union[str, Path] = DEFAULT_TEAM_PATH,
        battle_format: str = "gen9vgc2026regf",
        account_configuration: Optional[AccountConfiguration] = None,
        server_configuration: ServerConfiguration = LocalhostServerConfiguration,
        team: Optional[Union[str, Teambuilder]] = None,
        **kwargs
    ):
        """
        Initialize VGC Player.

        Args:
            team_path: Path to team file (showdown format)
            battle_format: VGC format string (default: gen9vgc2026regf)
            account_configuration: Player account config (auto-generated if None)
            server_configuration: Server config (default: localhost)
            team: Team string or Teambuilder (loaded from team_path if None)
            **kwargs: Additional arguments passed to Player
        """
        # Load team if not provided
        if team is None:
            team = load_team(team_path)

        # Generate random account if not provided
        if account_configuration is None:
            account_configuration = random_account("VGC")

        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            team=team,
            accept_open_team_sheet=True,  # Required for VGC
            server_configuration=server_configuration,
            **kwargs
        )

        self._team_path = team_path

    def teampreview(self, battle: DoubleBattle) -> str:
        """
        Handle VGC team preview by randomly selecting 4 from 6 Pokemon.

        Override this method to implement a learned team selection policy.

        Args:
            battle: The current battle state during team preview

        Returns:
            Team selection command string (e.g., "/team 1234")
        """
        # Get available team members (1-indexed)
        members = list(range(1, len(battle.team) + 1))

        # Randomly shuffle for selection
        random.shuffle(members)

        # Select up to max_team_size (4 for VGC doubles)
        if battle.max_team_size:
            members = members[:battle.max_team_size]

        return "/team " + "".join(str(m) for m in members)

    def choose_move(self, battle: DoubleBattle):
        """
        Choose a move for the current turn.

        This base implementation chooses randomly. Override in subclasses
        to implement RL or heuristic policies.

        Args:
            battle: The current battle state

        Returns:
            A BattleOrder for this turn
        """
        return self.choose_random_doubles_move(battle)

    @property
    def team_path(self) -> Path:
        """Path to the team file used by this player."""
        return Path(self._team_path)

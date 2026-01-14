"""
Random VGC Player for baseline opponent in training.

This player makes random (but legal) moves each turn, useful as a
first-stage opponent for RL training before self-play.
"""

import sys
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle import DoubleBattle
from poke_env.ps_client import AccountConfiguration, ServerConfiguration, LocalhostServerConfiguration

from .vgc_player import VGCPlayer, random_account, DEFAULT_TEAM_PATH


class RandomVGCPlayer(VGCPlayer):
    """
    VGC Player that makes random legal moves.

    This is a baseline opponent for initial RL training. It:
    - Properly handles VGC team preview (random 4-of-6 selection)
    - Accepts open team sheets
    - Chooses randomly from valid (joint-legal) orders each turn

    Example usage:
        opponent = RandomVGCPlayer(team_path="teams/team.txt")
        # Use as opponent in training environment
    """

    def __init__(
        self,
        team_path: Union[str, Path] = DEFAULT_TEAM_PATH,
        battle_format: str = "gen9vgc2026regf",
        account_configuration: Optional[AccountConfiguration] = None,
        server_configuration: ServerConfiguration = LocalhostServerConfiguration,
        **kwargs
    ):
        """
        Initialize Random VGC Player.

        Args:
            team_path: Path to team file
            battle_format: VGC format string
            account_configuration: Player account (auto-generated with "Random" prefix if None)
            server_configuration: Server config
            **kwargs: Additional arguments passed to VGCPlayer
        """
        if account_configuration is None:
            account_configuration = random_account("Random")

        super().__init__(
            team_path=team_path,
            battle_format=battle_format,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            **kwargs
        )

    def choose_move(self, battle: DoubleBattle):
        """
        Choose a random legal move.

        Uses poke-env's built-in random doubles move selection,
        which properly handles joint-legality constraints.

        Args:
            battle: The current battle state

        Returns:
            A random legal BattleOrder
        """
        return self.choose_random_doubles_move(battle)

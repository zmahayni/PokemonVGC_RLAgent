"""
Heuristic VGC Player that chooses moves based on type matchups and damage estimation.

This player evaluates each move's potential damage against opponent targets and
selects the highest-scoring move. It considers:
- Base power
- Type effectiveness
- STAB bonus
- Physical vs Special stat ratios
- Spread move bonuses
"""

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.target import Target
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from .vgc_player import VGCPlayer, DEFAULT_TEAM_PATH


# Spread move targets that hit multiple opponents
SPREAD_TARGETS = {
    Target.ALL_ADJACENT_FOES,
    Target.ALL_ADJACENT,
    Target.ALL,
}


class HeuristicVGCPlayer(VGCPlayer):
    """
    Heuristic VGC Player that chooses moves based on damage estimation.

    Evaluates moves using: base_power × type_effectiveness × STAB × stat_ratio × accuracy
    Prefers spread moves when both opponents are targetable.
    Falls back to random moves when no damaging options are available.
    """

    def __init__(
        self,
        team_path: Union[str, Path] = DEFAULT_TEAM_PATH,
        battle_format: str = "gen9vgc2026regf",
        **kwargs
    ):
        super().__init__(
            team_path=team_path,
            battle_format=battle_format,
            **kwargs
        )

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        """Choose moves based on damage estimation for both active Pokemon."""
        try:
            return self._choose_heuristic_move(battle)
        except Exception:
            # Any error in heuristic logic, fall back to random
            return self.choose_random_doubles_move(battle)

    def _choose_heuristic_move(self, battle: DoubleBattle) -> BattleOrder:
        """Internal heuristic move selection."""
        # Handle force switch situations
        if any(battle.force_switch):
            return self.choose_random_doubles_move(battle)

        orders: List[SingleBattleOrder] = []
        switched_in: Optional[Pokemon] = None
        used_heuristic = False

        for pos, (mon, moves, switches) in enumerate(zip(
            battle.active_pokemon,
            battle.available_moves,
            battle.available_switches
        )):
            # Filter out already-used switch target
            available_switches = [s for s in switches if s != switched_in]

            # Handle fainted/missing Pokemon
            if not mon or mon.fainted:
                orders.append(PassBattleOrder())
                continue

            # Handle no moves available
            if not moves:
                if available_switches:
                    switch_target = random.choice(available_switches)
                    orders.append(SingleBattleOrder(switch_target))
                    switched_in = switch_target
                else:
                    orders.append(PassBattleOrder())
                continue

            # Find best move/target combination
            best_order = self._find_best_move(battle, mon, moves, pos)

            if best_order is not None:
                orders.append(best_order)
                used_heuristic = True
            elif moves:
                # Fallback: pick first move with valid target
                move = moves[0]
                targets = battle.get_possible_showdown_targets(move, mon)
                if targets:
                    orders.append(SingleBattleOrder(move, move_target=targets[0]))
                else:
                    orders.append(SingleBattleOrder(move))
            elif available_switches:
                # No good attacking moves, switch to best matchup
                switch_target = self._find_best_switch(battle, available_switches)
                orders.append(SingleBattleOrder(switch_target))
                switched_in = switch_target
            else:
                orders.append(PassBattleOrder())

        # If we didn't use heuristic at all, fall back to random
        if not used_heuristic:
            return self.choose_random_doubles_move(battle)

        # Join orders respecting joint-legality constraints
        if len(orders) >= 2 and (orders[0] or orders[1]):
            joined = DoubleBattleOrder.join_orders([orders[0]], [orders[1]])
            if joined:
                return joined[0]

        return self.choose_random_doubles_move(battle)

    def _find_best_move(
        self,
        battle: DoubleBattle,
        mon: Pokemon,
        moves: List[Move],
        pos: int
    ) -> Optional[SingleBattleOrder]:
        """Find the best move and target for a given Pokemon."""
        best_move: Optional[Move] = None
        best_target: Optional[int] = None
        best_score: float = -1

        for move in moves:
            # Skip status moves (no damage)
            if move.base_power == 0:
                continue

            # Get possible targets
            targets = battle.get_possible_showdown_targets(move, mon)

            for target in targets:
                # Only consider opponent targets (1 and 2 in showdown notation)
                if target not in [battle.OPPONENT_1_POSITION, battle.OPPONENT_2_POSITION]:
                    continue

                # Get target Pokemon (target is 1-indexed, list is 0-indexed)
                target_idx = target - 1
                if target_idx < 0 or target_idx >= len(battle.opponent_active_pokemon):
                    continue

                opp = battle.opponent_active_pokemon[target_idx]
                if not opp or opp.fainted:
                    continue

                # Calculate damage score
                score = self._estimate_damage(move, mon, opp)

                # Bonus for spread moves that hit both opponents
                if move.target in SPREAD_TARGETS:
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
        """
        Estimate damage for a move.

        Formula: base_power × type_mult × STAB × stat_ratio × accuracy
        """
        base = move.base_power

        # Type effectiveness
        type_mult = defender.damage_multiplier(move)

        # STAB bonus (Same Type Attack Bonus)
        stab = 1.5 if move.type in attacker.types else 1.0

        # Stat ratio based on move category
        if move.category == MoveCategory.PHYSICAL:
            atk_stat = attacker.base_stats.get("atk", 100)
            def_stat = defender.base_stats.get("def", 100)
        elif move.category == MoveCategory.SPECIAL:
            atk_stat = attacker.base_stats.get("spa", 100)
            def_stat = defender.base_stats.get("spd", 100)
        else:
            # Status move
            return 0

        stat_ratio = atk_stat / max(def_stat, 1)

        # Accuracy factor
        accuracy = move.accuracy if move.accuracy else 1.0

        return base * type_mult * stab * stat_ratio * accuracy

    def _find_best_switch(
        self,
        battle: DoubleBattle,
        switches: List[Pokemon]
    ) -> Pokemon:
        """Find the best Pokemon to switch in based on type matchups."""
        if not switches:
            raise ValueError("No switches available")

        # Score each switch option based on matchup against active opponents
        best_switch = switches[0]
        best_score = float('-inf')

        for switch in switches:
            score = 0
            for opp in battle.opponent_active_pokemon:
                if opp and not opp.fainted:
                    score += self._estimate_matchup(switch, opp)

            if score > best_score:
                best_score = score
                best_switch = switch

        return best_switch

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """
        Estimate matchup advantage between two Pokemon.

        Positive = we have advantage, Negative = they have advantage
        """
        # How much damage can we do to them (type effectiveness)
        our_offense = max(
            [opponent.damage_multiplier(t) for t in mon.types if t is not None],
            default=1.0
        )

        # How much damage can they do to us
        their_offense = max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None],
            default=1.0
        )

        return our_offense - their_offense

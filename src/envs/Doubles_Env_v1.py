from poke_env.environment.doubles_env import DoublesEnv
import numpy as np
from gymnasium.spaces import Box


class Doubles_Env_v1(DoublesEnv):
    def __init__(self, battle_format, team=None):
        super().__init__(
            battle_format=battle_format,
            save_replays=False,
            strict=False,
            fake=False,
            start_timer_on_battle_start=True,
            accept_open_team_sheet=False,
            team=team,
        )

        # Observation layout:
        # my2 HP (2), opp2 HP (2), then for ally1: base power (4), type eff (4), accuracy (4),
        # then for ally2: base power (4), type eff (4), accuracy (4).
        # -> shape (4 + 12 + 12) = (28,)
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(28,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def _safe_hp_fraction(self, p):
        try:
            return p.current_hp_fraction if p is not None else 0.0
        except Exception:
            return 0.0

    def embed_battle(self, battle):
        # Access allied and opponent actives by position when available
        ally1 = None
        ally2 = None
        try:
            # poke-env Doubles uses indexing: battle.active_pokemon[pos]
            if hasattr(battle, "active_pokemon"):
                ap = battle.active_pokemon
                ally1 = ap[0] if isinstance(ap, (list, tuple)) and len(ap) > 0 else ap
                ally2 = ap[1] if isinstance(ap, (list, tuple)) and len(ap) > 1 else None
        except Exception:
            pass

        # Fallbacks in case of different attribute names
        if ally1 is None:
            ally1 = getattr(battle, "active_pokemon", None)
        if ally2 is None:
            ally2 = getattr(battle, "active_pokemon2", None) or getattr(
                battle, "active_pokemon_2", None
            )

        opp1 = None
        opp2 = None
        try:
            if hasattr(battle, "opponent_active_pokemon"):
                op = battle.opponent_active_pokemon
                opp1 = op[0] if isinstance(op, (list, tuple)) and len(op) > 0 else op
                opp2 = op[1] if isinstance(op, (list, tuple)) and len(op) > 1 else None
        except Exception:
            pass
        if opp1 is None:
            opp1 = getattr(battle, "opponent_active_pokemon", None)
        if opp2 is None:
            opp2 = getattr(battle, "opponent_active_pokemon2", None) or getattr(
                battle, "opponent_active_pokemon_2", None
            )

        hp = np.zeros((4,), dtype=np.float32)
        hp[0] = self._safe_hp_fraction(ally1)
        hp[1] = self._safe_hp_fraction(ally2)
        hp[2] = self._safe_hp_fraction(opp1)
        hp[3] = self._safe_hp_fraction(opp2)

        def move_features_for_pos(pos, ally_pokemon):
            bp = np.zeros((4,), dtype=np.float32)
            te = np.zeros((4,), dtype=np.float32)
            acc = np.zeros((4,), dtype=np.float32)

            if ally_pokemon is None:
                return bp, te, acc

            # Construct moves list per poke-env doubles rules
            try:
                moves = []
                if (
                    hasattr(battle, "available_moves")
                    and isinstance(battle.available_moves, (list, tuple))
                    and len(battle.available_moves) > pos
                    and len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                ):
                    moves = battle.available_moves[pos]
                else:
                    moves = list(ally_pokemon.moves.values())
            except Exception:
                moves = []

            if len(moves) > 4:
                moves = moves[0:4]

            # Use first opponent for type effectiveness context by default
            opp_for_te = opp1

            for i, m in enumerate(moves):
                try:
                    base_power = float(m.base_power) if m.base_power is not None else 0.0
                    base_power = min(base_power, 150.0)
                    bp[i] = base_power / 150.0
                except Exception:
                    bp[i] = 0.0

                try:
                    if opp_for_te is not None and m.type is not None:
                        type_effectiveness = min(
                            opp_for_te.damage_multiplier(m.type), 4.0
                        )
                        te[i] = type_effectiveness / 4.0
                    else:
                        te[i] = 0.0
                except Exception:
                    te[i] = 0.0

                try:
                    accuracy = m.accuracy
                    acc[i] = 1.0 if not accuracy else float(accuracy)
                except Exception:
                    acc[i] = 1.0

            return bp, te, acc

        bp1, te1, acc1 = move_features_for_pos(0, ally1)
        bp2, te2, acc2 = move_features_for_pos(1, ally2)

        obs = np.concatenate((hp, bp1, te1, acc1, bp2, te2, acc2), axis=0)
        return obs

    def calc_reward(self, battle):
        # Same heuristic as singles for now
        return self.reward_computing_helper(
            battle, fainted_value=1.0, hp_value=0.5, status_value=0.1, victory_value=1.0
        )

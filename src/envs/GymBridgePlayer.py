import numpy as np
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder

import asyncio

class GymBridgePlayer(Player):
    """
    A tiny Player that lets *you* choose the move by setting a pending action
    (integer 0..3). In choose_move(), we map that slot to an actual move.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending_action = None  # set by your gym env before a turn
        self.action_ready = asyncio.Event()

    # Your gym env calls this before advancing a turn
    def set_pending_action(self, action_idx: int):
        self.pending_action = int(action_idx)
        self.action_ready.set()

    async def choose_move(self, battle):
        """
        poke-env calls this every decision. We:
        1) Build a stable 4-slot move list from the active Pokémon
        2) If the chosen slot is usable, play it
        3) Otherwise fall back to the best legal move (e.g., highest bp * type eff)
        """
        # 1) Make a stable 4-slot view of the active's known moves
        #    (active_pokemon.moves is a dict of Move objects keyed by move id)

        if not battle.available_moves and battle.available_switches:
            switch = battle.available_switches[-1]
            return self.create_order(switch)
    
        await self.action_ready.wait()

        known_moves = list(battle.active_pokemon.moves.values())  # may be < 4 early
        slots = [known_moves[i] if i < len(known_moves) else None for i in range(4)]

        # Helper: check if a move is usable *this turn*
        def usable(m):
            if m is None:
                return False
            # PP must be positive and move must be in battle.available_moves this turn
            if m.current_pp is not None and m.current_pp <= 0:
                return False
            return any(m.id == am.id for am in battle.available_moves)

        # 2) If we have a pending action and it’s usable, play it
        if self.pending_action is not None:
            idx = int(self.pending_action)
            self.pending_action = None  # consume it
            self.action_ready.clear()
            if 0 <= idx < 4 and usable(slots[idx]):
                return self.create_order(slots[idx])

        # 3) Fallback: pick the “best” legal move among available ones
        #    Score = base_power * type_effectiveness (very rough but fine for v1)
        best = None
        best_score = -1.0

        # build foe typing (may be None very early; default neutral multiplier)
        foe = battle.opponent_active_pokemon
        foe_types = set(foe.types) if foe is not None and foe.types is not None else set()

        for m in battle.available_moves:
            bp = float(m.base_power or 0.0)
            # crude type effectiveness: multiply effectiveness vs each foe type
            mult = 1.0
            if m.type is not None and foe_types:
                for t in foe_types:
                    try:
                        mult *= m.type.damage_multiplier(t)
                    except Exception:
                        pass
            score = bp * mult
            if score > best_score:
                best_score = score
                best = m

        if best is not None:
            return self.create_order(best)

        # If truly nothing is available (edge cases), let poke-env pick something valid
        return self.choose_random_move(battle)

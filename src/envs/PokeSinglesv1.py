import gymnasium as gym
from gymnasium import spaces
import numpy as np
from poke_env.player import RandomPlayer
from .GymBridgePlayer import GymBridgePlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
import asyncio

# Format: gen9randombattle (both sides random; fixed team comes later in OU).

# Action space: Discrete(4) (moves only; no switching in v1).

# Obs vector (length ~10): 4×base power (normalized), 4×type effectiveness (normalized), my HP%, opp HP%.

# Rewards: 0 per turn; +1 win, −1 loss; truncate at 50 turns → 0.

# Mask: mark illegal move slots (no PP/disabled/empty/recharge) as not selectable; if the agent picks one, remap to the best legal.


class PokeSinglesV1(gym.Env):
    def __init__(self, max_turns=50, battle_format="gen9randombattle"):
        super().__init__()

        # 4 base powers, 4 type mults, my_hp%, opp_hp%
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        # Only 4 moves, ignoring switching for now
        self.action_space = spaces.Discrete(4)

        self.max_turns = max_turns
        self.battle_format = battle_format
        self.battle = None
        self.agent = GymBridgePlayer(
            battle_format="gen9randombattle",
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration.generate("agent_key", rand=True),
        )
        self.opponent = RandomPlayer(
            battle_format="gen9randombattle",
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration.generate("opp_key", rand=True),
        )

    def reset(self, seed=None, battle_format="gen9randombattle"):
        self.my_hp = 1
        self.opp_hp = 1
        self.turns = 0
        asyncio.self.agent.battle_against(self.opponent, n_battles=1)
        self.battle = list(self.agent.battles.values())[-1]
        obs = self.embed_observation(self.battle)
        info = {}

        return obs, info

    def step(self, action):
        self.turns += 1
        mask = self.action_mask(self.battle)
        if not mask[action]:
            action = self.remap_action(mask, self.battle)

        self.agent.set_pending_action(action)
        terminated = self.battle.finished
        truncated = self.turns >= self.max_turns
        reward = 0.0

        if truncated:
            reward = 0
        if terminated:  # not sure how to check this in showdown
            reward = 1 if self.battle.won else -1
        obs = self.embed_observation(self.battle)
        info = {}

        return obs, float(reward), terminated, truncated, info

    def action_mask(self, battle) -> np.ndarray:
        mask = np.zeros(4, dtype=bool)

        known = list(battle.active_pokemon.moves.values())
        slots = [known[i] if i < len(known) else None for i in range(4)]

        avail_ids = {m.id for m in battle.available_moves}
        for k, m in enumerate(slots):
            if m is None:
                mask[k] = False
            else:
                has_pp = (m.current_pp is None) or (m.current_pp > 0)
                mask[k] = has_pp and (m.id in avail_ids)

        if not mask.any():
            for k, m in enumerate(slots):
                if m is not None:
                    mask[k] = True
                    break
        return mask

    def embed_observation(self, battle) -> np.ndarray:
        bp = np.zeros(4, dtype=np.float32)
        te = np.zeros(4, dtype=np.float32)

        # stable 4-slot list
        known = list(battle.active_pokemon.moves.values())
        slots = [known[i] if i < len(known) else None for i in range(4)]

        # foe typing (may be None very early)
        foe = battle.opponent_active_pokemon
        foe_types = set(foe.types) if (foe and foe.types) else set()

        for k, m in enumerate(slots):
            if m is None:
                continue
            # base power normalized
            bp[k] = min(float(m.base_power or 0.0), 150.0) / 150.0

        # type effectiveness normalized (clip [0,4] then /4)
        mult = 1.0
        if m.type and foe_types:
            for t in foe_types:
                try:
                    mult *= m.type.damage_multiplier(t)
                except Exception:
                    pass
        te[k] = min(max(mult, 0.0), 4.0) / 4.0

        my_hp = float(battle.active_pokemon.current_hp_fraction or 0.0)
        op_hp = float(battle.opponent_active_pokemon.current_hp_fraction or 0.0)

        obs = np.concatenate([bp, te, [my_hp], [op_hp]]).astype(np.float32)
        return obs

    def remap_action(self, mask: np.ndarray, battle) -> int:
        best_idx = 0
        best_score = -1.0

        # Stable 4 slots again
        known = list(battle.active_pokemon.moves.values())
        slots = [known[i] if i < len(known) else None for i in range(4)]

        foe = battle.opponent_active_pokemon
        foe_types = set(foe.types) if foe and foe.types else set()

        for k, m in enumerate(slots):
            if not mask[k] or m is None:
                continue
            bp = float(m.base_power or 0.0)
            mult = 1.0
            if m.type and foe_types:
                for t in foe_types:
                    try:
                        mult *= m.type.damage_multiplier(t)
                    except Exception:
                        pass
            score = bp * mult
            if score > best_score:
                best_score = score
                best_idx = k

        return best_idx


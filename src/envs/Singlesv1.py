import gymnasium as gym
from gymnasium import spaces
import numpy as np
from GymBridgePlayer import GymBridgePlayer
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.battle.move_category import MoveCategory

class Singlesv1(gym.Env):
    def __init__(self):
        super().__init__()

        # base power * type effectivess per each move, %hp, opp %hp
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # index per each move
        self.action_space = spaces.Discrete(4)

        self.max_turns = 50
        self.battle_format = "gen9randombattle"
        self.agent = GymBridgePlayer(
            battle_format=self.battle_format,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration("Agent", None),
        )
        self.opponent = RandomPlayer(
            battle_format=self.battle_format,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration("Opp", None),
        )

        self.battle = None

    def reset(self):
        self.my_hp = 1
        self.opp_hp = 1
        self.turns = 0

        while True:
            if len(self.battle.available_moves) > 0 or (not self.battle.available_moves and self.battle.available_switches):
                break
            time.sleep(0.005)

        obs = self.embed_observation(self.battle)  # need to make self.battle first
        info = {}

        return obs, info

    def step(self, action):
        t0 = self.battle.turn
        self.turns += 1

        # if forced switch
        if not self.battle.available_moves and self.battle.available_switches:
            # switch to first pokemon
            

        moves = []
        for move in list(self.battle.active_pokemon.moves.values()):
            moves.append(move)

        # if chosen move is not legal, do random move
        if moves[action] not in self.battle.available_moves:
            # TODO: send action to GymBridgePlayer
            return self.battle.available_moves[0] #i know there is supposed to be a function, will make later
            


            # TODO: send action to GymBridgePlayer

        while self.battle.turn == t0 and not self.battle.finished:
            time.sleep(0.005)
            continue
    
        
        terminated = self.battle.finished
        truncated = self.turns >= self.max_turns
        reward = 0.0

        if truncated:
            reward = 0
        if terminated:
            reward = 1 if self.battle.won else -1

        obs = self.embed_observation(self.battle)
        info = {}

        return obs, float(reward), terminated, truncated, info

    def embed_observation(self, battle):
        power = np.zeros(4, dtype=np.float32)

        moves = []
        for move in list(battle.active_pokemon.moves.values()):
            moves.append(move)
        while len(moves) < 4:
            moves.append(None)

        for i, m in enumerate(moves):
            if m.category == MoveCategory.STATUS:
                power[i] = 0
            else:
                base_power = min(150,int(m.base_power)) if m else 0
                mult = 1
                opponent_types = battle.opponent_active_pokemon.types
                for t in opponent_types:
                    mult *= m.type.damage_multiplier(t)
                # normalize power to 0-1
                power[i] = (base_power * mult) / 600

        hp = float(battle.active_pokemon.current_hp_fraction)
        opp_hp = float(battle.opponent_active_pokemon.current_hp_fraction)

        obs = np.concatenate([power, [hp], [opp_hp]]).astype(np.float32)

        return obs

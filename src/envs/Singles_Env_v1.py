from poke_env.environment.singles_env import SinglesEnv
import numpy as np
from gymnasium.spaces import Box

class Singles_Env_v1(SinglesEnv):
    
    def __init__(self):
        super().__init__(battle_format='gen8randombattle', save_replays=False, strict=False, accept_open_team_sheet=False)

        #my_hp, opp_hp, 4base powers, 4type effectivness 4 accuracies -> shape is (14,)
        self.observation_spaces = {agent: Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32) for agent in self.possible_agents}

    def embed_battle(self, battle):

        hp = np.zeros((2,), dtype=np.float32)
        #set my_hp and opp_hp
        hp[0] = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        hp[1] = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0

        moves = []
        #if we can only struggle/recharge, then move list is only struggle/recharge
        if len(battle.available_moves) == 1 and battle.available_moves[0].id in ['struggle', 'recharge']:
            moves = battle.available_moves
        else:
            moves = list(battle.active_pokemon.moves.values())
        
        #bound to 4 moves just in case
        if len(moves) > 4:
            moves = moves[0:4]

        bp = np.zeros((4,), dtype=np.float32)
        #fill base powers
        for i, m in enumerate(moves):
            base_power = min(m.base_power, 150.0)
            base_power_norm = base_power/150.0
            bp[i] = base_power_norm

        te = np.zeros((4,), dtype=np.float32)
        #fill type effectiveness
        for i, m in enumerate(moves):
            type_effectiveness = min(battle.opponent_active_pokemon.damage_multiplier(m.type), 2.0)
            type_effectiveness_norm = type_effectiveness/2.0
            te[i] = type_effectiveness_norm
        
        acc = np.zeros((4,), dtype=np.float32)
        #fill accuracy
        for i, m in enumerate(moves):
            accuracy = min(m.accuracy, 100.0)
            acc[i] = accuracy

        obs = np.concatenate((hp, bp, te, acc), axis=0)

        return obs

    def calc_reward(self, battle):

        return self.reward_computing_helper(battle, fainted_value=1.0, hp_value=0.5, status_value=0.1, victory_value=1.0)







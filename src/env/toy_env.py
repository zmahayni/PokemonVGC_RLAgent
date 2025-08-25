from gymnasium.spaces import Space, Box, Discrete
import numpy as np

class Player():
    def __init__(self):
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) 
        self.action_space = Discrete(2)
        self.max_hp = 10
        self.max_turns = 100
        self.last_adv = 0.0
    
    def reset(self):
        self.hp = self.max_hp
        self.opp_hp = self.max_hp
        self.turn_counter = 0
        self.last_adv = 0.0

        obs = np.array([self.hp/self.max_hp, self.opp_hp/self.max_hp], dtype=np.float32)
        info = {}

        return obs, info

    def step(self, action):
        player_attack = (action == 0)
        player_defend = (action == 1)

        reward = 0

        terminated = False
        truncated = False
        info = {}

        opp_attack = True
        opp_defend = False

        player_damage = 1 if player_attack else 0
        opp_damage = 1 if opp_attack else 0

        incoming_damage = opp_damage
        if player_defend:
            incoming_damage //= 2
        
        incoming_to_opp = player_damage

        self.hp = max(0, self.hp - incoming_damage)
        self.opp_hp = max(0, self.opp_hp - incoming_to_opp)
        self.turn_counter += 1

        adv = (self.hp - self.opp_hp) / float(self.max_hp)
        reward = adv - self.last_adv
        self.last_adv = adv


        if self.opp_hp == 0 and self.hp > 0:
            reward += 1
            terminated = True
        elif self.hp == 0 and self.opp_hp > 0:
            reward -=1
            terminated = True
        elif self.hp == 0 and self.opp_hp == 0:
            terminated = True
        elif self.turn_counter >= self.max_turns:
            truncated = True
        

        obs = np.array([self.hp/self.max_hp, self.opp_hp/self.max_hp], dtype=np.float32)
        
        return obs, float(reward), terminated, truncated, info
        
            
            
                

        

        
        


    

        

    
    





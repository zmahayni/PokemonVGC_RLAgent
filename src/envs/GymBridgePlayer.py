import numpy as np
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder

import asyncio

class GymBridgePlayer(Player):
    def __init__(self):
        self.pending_action = None
        self.action_ready = asyncio.Event.clear
    
    def set_pending_action(self, idx):
        self.pending_action = idx
        self.action_ready.set()

    await def choose_move(self):
        if not self.battle.available_moves and self.battle.available_switches:
            switch = self.battle.available_switches[0]
            return create_order(switch)
        
        await self.action_ready.wait()


    

    
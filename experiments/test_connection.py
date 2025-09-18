import asyncio
import logging
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
from src.envs.GymBridgePlayer import GymBridgePlayer

async def main():
    agent = GymBridgePlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration.generate("agent_key", rand=True),
        log_level=logging.INFO,
    )
    opp = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration.generate("opp_key", rand=True),
        log_level=logging.INFO,
    )
    await agent.battle_against(opp, n_battles=1)
    print("Finished:", agent.n_finished_battles, "Won:", agent.n_won_battles)

if __name__ == "__main__":
    asyncio.run(main())

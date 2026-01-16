# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement Learning research project for Pokemon VGC (Video Game Championships) Doubles battles. Focus is on building reliable environments and experimenting with different RL approaches, eventually creating an agent that can play decently. The current plan is to have it be good with one team, and have it play games against other meta teams, then having it play against me. Eventually, it will be put on the official ladder. 

You are to serve as a research partner. Whenever we make a plan together, put that document in a folder called plans and include the date. Don't put (Recommended) when giving options in plan mode, just put the options themselves.

**Stack:** Stable Baselines3, poke-env, Pokemon Showdown, PyTorch, Gymnasium

## Common Commands

```bash
# Start Pokemon Showdown server (required before running anything)
cd pokemon-showdown && node pokemon-showdown start --no-security

# Exploration scripts (Phase 2 - understanding poke-env)
python src/explore/test_connection.py      # Test server connectivity
python src/explore/explore_battle_state.py # Explore DoubleBattle state
python src/explore/explore_actions.py      # Explore action space encoding
```
Always go into venv before running python commands

## Project Structure

```
src/
├── explore/                    # Exploration/debugging scripts
│   ├── test_connection.py      # Verify poke-env + Showdown connectivity
│   ├── explore_battle_state.py # Print DoubleBattle state each turn
│   └── explore_actions.py      # Document action encoding and joint-legality
teams/                          # VGC team files (user-provided)
configs/                        # Hyperparameter configs (to be created)
poke-env/                       # Local poke-env library (not PyPI)
pokemon-showdown/               # Local game server
```

## poke-env Doubles Action Space

**MultiDiscrete([107, 107])** - one action per active Pokemon position

Per-position encoding:
- `-2`: default order
- `-1`: forfeit
- `0`: pass
- `1-6`: switch to team member 1-6
- `7+`: moves with targets and gimmicks

Move formula: `action = 7 + (move_idx * 5) + target_offset + (gimmick * 20)`
- `move_idx`: 0-3 (4 moves)
- `target_offset`: 0-4 (targets -2 to +2)
- `gimmick`: 0=none, 1=mega, 2=z-move, 3=dynamax, 4=tera

Target positions:
- `-2`: ally at slot 0
- `-1`: ally at slot 1 / self
- `0`: no target (spread moves)
- `1`: opponent at slot 0
- `2`: opponent at slot 1

## Key poke-env Files

- `poke-env/src/poke_env/environment/doubles_env.py` - DoublesEnv, action_to_order/order_to_action
- `poke-env/src/poke_env/battle/double_battle.py` - DoubleBattle state, valid_orders
- `poke-env/src/poke_env/player/battle_order.py` - DoubleBattleOrder.join_orders() for joint-legality

## Joint-Legality Constraints

Not all action pairs are valid in doubles. `DoubleBattleOrder.join_orders()` checks:
- Both Pokemon can't use the same gimmick (tera, mega, dynamax, z-move)
- Both Pokemon can't switch to the same team member
- Both Pokemon can't pass (at least one must act)

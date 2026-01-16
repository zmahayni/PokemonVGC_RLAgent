"""
Battle feature extraction utilities for VGC environment observations.

Extracts meaningful features from DoubleBattle state for RL training.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "poke-env" / "src"))

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field


# Stat boost keys in order
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Status conditions we track (excluding FNT which is handled by HP=0)
STATUS_CONDITIONS = [Status.PAR, Status.BRN, Status.PSN, Status.TOX, Status.SLP, Status.FRZ]

# Weather types we track
WEATHER_TYPES = [Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOWSCAPE]

# Terrain types we track
TERRAIN_TYPES = [Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN]


def extract_pokemon_hp(pokemon: Optional[Pokemon]) -> float:
    """Extract HP fraction from a Pokemon (0.0 if None or fainted)."""
    if pokemon is None:
        return 0.0
    return pokemon.current_hp_fraction


def extract_stat_boosts(pokemon: Optional[Pokemon]) -> npt.NDArray[np.float32]:
    """
    Extract stat boosts as normalized values in [-1, 1].

    Boosts range from -6 to +6, so we divide by 6.
    Returns array of shape (7,) for [atk, def, spa, spd, spe, accuracy, evasion].
    """
    if pokemon is None:
        return np.zeros(len(BOOST_STATS), dtype=np.float32)

    boosts = pokemon.boosts
    return np.array([boosts.get(stat, 0) / 6.0 for stat in BOOST_STATS], dtype=np.float32)


def extract_status(pokemon: Optional[Pokemon]) -> npt.NDArray[np.float32]:
    """
    Extract status conditions as one-hot encoding.

    Returns array of shape (6,) for [PAR, BRN, PSN, TOX, SLP, FRZ].
    Each value is 1.0 if the Pokemon has that status, 0.0 otherwise.
    """
    if pokemon is None:
        return np.zeros(len(STATUS_CONDITIONS), dtype=np.float32)

    status = pokemon.status
    return np.array([1.0 if status == s else 0.0 for s in STATUS_CONDITIONS], dtype=np.float32)


def extract_weather(battle: DoubleBattle) -> npt.NDArray[np.float32]:
    """
    Extract current weather as one-hot encoding.

    Returns array of shape (4,) for [SUN, RAIN, SAND, SNOW].
    """
    weather_dict = battle.weather
    result = np.zeros(len(WEATHER_TYPES), dtype=np.float32)

    for i, weather_type in enumerate(WEATHER_TYPES):
        if weather_type in weather_dict:
            result[i] = 1.0

    return result


def extract_terrain(battle: DoubleBattle) -> npt.NDArray[np.float32]:
    """
    Extract current terrain as one-hot encoding.

    Returns array of shape (4,) for [ELECTRIC, GRASSY, MISTY, PSYCHIC].
    """
    fields = battle.fields
    result = np.zeros(len(TERRAIN_TYPES), dtype=np.float32)

    for i, terrain_type in enumerate(TERRAIN_TYPES):
        if terrain_type in fields:
            result[i] = 1.0

    return result


def extract_trick_room(battle: DoubleBattle) -> float:
    """Check if Trick Room is active."""
    return 1.0 if Field.TRICK_ROOM in battle.fields else 0.0


def extract_type_matchup_hints(battle: DoubleBattle) -> npt.NDArray[np.float32]:
    """
    Extract hints about type matchups.

    For each of our active Pokemon, check if any of its moves are super effective
    against any opponent Pokemon.

    Returns array of shape (4,):
    - [0]: Our Pokemon 0 has SE move vs opponent 0
    - [1]: Our Pokemon 0 has SE move vs opponent 1
    - [2]: Our Pokemon 1 has SE move vs opponent 0
    - [3]: Our Pokemon 1 has SE move vs opponent 1
    """
    result = np.zeros(4, dtype=np.float32)

    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    for i, our_mon in enumerate(our_active[:2]):
        if our_mon is None:
            continue

        # Get our Pokemon's available moves
        if i < len(battle.available_moves) and battle.available_moves[i]:
            moves = battle.available_moves[i]
        else:
            moves = list(our_mon.moves.values()) if our_mon.moves else []

        for j, opp_mon in enumerate(opp_active[:2]):
            if opp_mon is None:
                continue

            # Check if any move is super effective
            for move in moves:
                if move.type is not None and opp_mon.type_1 is not None:
                    multiplier = move.type.damage_multiplier(
                        opp_mon.type_1,
                        opp_mon.type_2,
                        type_chart=battle._data.type_chart
                    )
                    if multiplier >= 2.0:
                        result[i * 2 + j] = 1.0
                        break

    return result


def extract_speed_comparison(battle: DoubleBattle) -> npt.NDArray[np.float32]:
    """
    Compare speeds between our and opponent's active Pokemon.

    Returns array of shape (2,):
    - [0]: 1.0 if our Pokemon 0 is faster than both opponents, else 0.0
    - [1]: 1.0 if our Pokemon 1 is faster than both opponents, else 0.0

    Note: This is a simplified comparison - doesn't account for boosts, items, etc.
    """
    result = np.zeros(2, dtype=np.float32)

    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # Get opponent speeds
    opp_speeds = []
    for opp_mon in opp_active[:2]:
        if opp_mon is not None and opp_mon.stats and "spe" in opp_mon.stats:
            opp_speeds.append(opp_mon.stats["spe"] or 0)
        else:
            opp_speeds.append(0)

    max_opp_speed = max(opp_speeds) if opp_speeds else 0

    # Compare our speeds
    for i, our_mon in enumerate(our_active[:2]):
        if our_mon is not None and our_mon.stats and "spe" in our_mon.stats:
            our_speed = our_mon.stats["spe"] or 0
            if our_speed > max_opp_speed:
                result[i] = 1.0

    return result


def extract_all_features(battle: DoubleBattle) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Extract all features from a battle state.

    Returns a dictionary with all feature arrays for easy debugging and assembly.
    """
    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # Get bench Pokemon (non-active, non-fainted)
    our_bench = [m for m in battle.team.values() if not m.active and not m.fainted][:2]
    opp_bench = [m for m in battle.opponent_team.values() if not m.active and not m.fainted][:2]

    features = {
        # HP features
        "our_active_hp": np.array([extract_pokemon_hp(p) for p in our_active[:2]], dtype=np.float32),
        "our_bench_hp": np.array([extract_pokemon_hp(our_bench[i]) if i < len(our_bench) else 0.0
                                  for i in range(2)], dtype=np.float32),
        "opp_active_hp": np.array([extract_pokemon_hp(p) for p in opp_active[:2]], dtype=np.float32),
        "opp_bench_hp": np.array([extract_pokemon_hp(opp_bench[i]) if i < len(opp_bench) else 0.0
                                  for i in range(2)], dtype=np.float32),

        # Stat boosts (7 stats x 2 Pokemon = 14 per side)
        "our_boosts": np.concatenate([extract_stat_boosts(p) for p in our_active[:2]]),
        "opp_boosts": np.concatenate([extract_stat_boosts(p) for p in opp_active[:2]]),

        # Status conditions (6 conditions x 2 Pokemon = 12 per side)
        "our_status": np.concatenate([extract_status(p) for p in our_active[:2]]),
        "opp_status": np.concatenate([extract_status(p) for p in opp_active[:2]]),

        # Weather and terrain
        "weather": extract_weather(battle),
        "terrain": extract_terrain(battle),
        "trick_room": np.array([extract_trick_room(battle)], dtype=np.float32),

        # Type matchups and speed
        "type_matchup": extract_type_matchup_hints(battle),
        "speed_advantage": extract_speed_comparison(battle),

        # Gimmick availability
        "can_tera": np.array([float(battle.can_tera[i]) if i < len(battle.can_tera) else 0.0
                             for i in range(2)], dtype=np.float32),
        "can_mega": np.array([float(battle.can_mega_evolve[i]) if i < len(battle.can_mega_evolve) else 0.0
                             for i in range(2)], dtype=np.float32),
        "can_dynamax": np.array([float(battle.can_dynamax[i]) if i < len(battle.can_dynamax) else 0.0
                                for i in range(2)], dtype=np.float32),
        "can_zmove": np.array([float(battle.can_z_move[i]) if i < len(battle.can_z_move) else 0.0
                              for i in range(2)], dtype=np.float32),

        # Force switch and turn
        "force_switch": np.array([float(battle.force_switch[i]) if i < len(battle.force_switch) else 0.0
                                  for i in range(2)], dtype=np.float32),
        "turn": np.array([min(battle.turn / 50.0, 1.0)], dtype=np.float32),
    }

    return features


def features_to_observation(features: Dict[str, npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
    """
    Concatenate all features into a single observation vector.

    Total dimensions: 79
    - HP: 2 + 2 + 2 + 2 = 8
    - Boosts: 14 + 14 = 28
    - Status: 12 + 12 = 24
    - Weather: 4
    - Terrain: 4
    - Trick Room: 1
    - Type matchup: 4
    - Speed: 2
    - Gimmicks: 8
    - Force switch: 2
    - Turn: 1
    = 86 total (slightly more than planned due to including speed/trick room)
    """
    return np.concatenate([
        features["our_active_hp"],      # 2
        features["our_bench_hp"],       # 2
        features["opp_active_hp"],      # 2
        features["opp_bench_hp"],       # 2
        features["our_boosts"],         # 14
        features["opp_boosts"],         # 14
        features["our_status"],         # 12
        features["opp_status"],         # 12
        features["weather"],            # 4
        features["terrain"],            # 4
        features["trick_room"],         # 1
        features["type_matchup"],       # 4
        features["speed_advantage"],    # 2
        features["can_tera"],           # 2
        features["can_mega"],           # 2
        features["can_dynamax"],        # 2
        features["can_zmove"],          # 2
        features["force_switch"],       # 2
        features["turn"],               # 1
    ])


# Observation dimension for the v2 environment
OBS_DIM_V2 = 86

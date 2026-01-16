"""
VGC Environment implementations for RL training.

Three environment variants are available:

1. VGCEnvDiscrete (default as VGCEnv):
   - Action space: Discrete(11449) = 107*107 flattened joint actions
   - Properly enforces joint-legality constraints via action masking
   - Use with MaskablePPO for best results

2. VGCEnvMultiDiscrete:
   - Action space: MultiDiscrete([107, 107])
   - Each Pokemon selects independently from 107 actions
   - Simpler but doesn't enforce joint constraints at action level

3. VGCEnvV2:
   - Same action space as VGCEnvDiscrete (Discrete(11449))
   - Improved observations (86 features vs 64):
     * Stat stages, status conditions, weather/terrain, type matchups
     * Fixed bench count (2 instead of 4)
   - Enhanced rewards: status infliction, stat boosts, speed control
"""

from .vgc_env_discrete import VGCEnvDiscrete
from .vgc_env_multidiscrete import VGCEnvMultiDiscrete
from .vgc_env_v2 import VGCEnvV2

# Default to Discrete version (recommended)
VGCEnv = VGCEnvDiscrete

__all__ = ["VGCEnv", "VGCEnvDiscrete", "VGCEnvMultiDiscrete", "VGCEnvV2"]

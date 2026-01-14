"""
Test script to verify the VGC RL setup works correctly.

This script runs through:
1. Player creation and team loading
2. Action masking utilities
3. Environment initialization
4. A few manual steps through the environment

Prerequisites:
    Start Pokemon Showdown server:
    cd pokemon-showdown && node pokemon-showdown start --no-security

Usage:
    python src/test_setup.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "poke-env" / "src"))

import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from src.players import VGCPlayer, RandomVGCPlayer
        print("  [OK] Players imported")
    except ImportError as e:
        print(f"  [FAIL] Players import error: {e}")
        return False

    try:
        from src.utils import get_action_mask, decode_action, ACTION_SPACE_SIZE
        print(f"  [OK] Utils imported (ACTION_SPACE_SIZE={ACTION_SPACE_SIZE})")
    except ImportError as e:
        print(f"  [FAIL] Utils import error: {e}")
        return False

    try:
        from src.envs import VGCEnv
        print("  [OK] Environment imported")
    except ImportError as e:
        print(f"  [FAIL] Environment import error: {e}")
        return False

    print("\nAll imports successful!")
    return True


def test_team_loading():
    """Test team loading."""
    print("\n" + "=" * 60)
    print("Testing team loading...")
    print("=" * 60)

    from src.players.vgc_player import load_team, DEFAULT_TEAM_PATH

    try:
        team = load_team(DEFAULT_TEAM_PATH)
        print(f"  [OK] Team loaded from {DEFAULT_TEAM_PATH}")
        print(f"  [OK] Team has {team.count('|')} lines")
        return True
    except Exception as e:
        print(f"  [FAIL] Team loading error: {e}")
        return False


def test_action_decoding():
    """Test action decoding utility."""
    print("\n" + "=" * 60)
    print("Testing action decoding...")
    print("=" * 60)

    from src.utils import decode_action

    test_cases = [
        (0, "pass"),
        (1, "switch to mon 1"),
        (6, "switch to mon 6"),
        (7, "move1 -> ally0"),
        (12, "move2 -> ally0"),
        (87, "move1 -> ally0+tera"),
    ]

    all_passed = True
    for action, expected in test_cases:
        result = decode_action(action)
        passed = expected in result
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Action {action}: {result}")
        if not passed:
            all_passed = False

    return all_passed


def test_environment_creation():
    """Test environment can be created."""
    print("\n" + "=" * 60)
    print("Testing environment creation...")
    print("=" * 60)

    from src.envs import VGCEnv

    try:
        env = VGCEnv()
        print(f"  [OK] Environment created")
        print(f"  [OK] Action space: {env.action_space}")
        print(f"  [OK] Observation space: {env.observation_space}")
        env.close()
        return True
    except Exception as e:
        print(f"  [FAIL] Environment creation error: {e}")
        return False


def test_environment_reset():
    """Test environment reset (requires server)."""
    print("\n" + "=" * 60)
    print("Testing environment reset (requires server)...")
    print("=" * 60)

    from src.envs import VGCEnv

    try:
        env = VGCEnv()
        print("  [INFO] Attempting reset (waiting for server connection)...")
        obs, info = env.reset()
        print(f"  [OK] Reset successful!")
        print(f"  [OK] Observation shape: {obs.shape}")
        print(f"  [OK] Battle tag: {info.get('battle_tag', 'N/A')}")
        print(f"  [OK] Action mask shape: {info['action_mask'].shape}")
        print(f"  [OK] Valid actions pos0: {info['action_mask'][:107].sum()}/107")
        print(f"  [OK] Valid actions pos1: {info['action_mask'][107:].sum()}/107")

        env.close()
        return True
    except RuntimeError as e:
        if "server" in str(e).lower() or "timeout" in str(e).lower():
            print(f"  [SKIP] Server not running: {e}")
            print("  [INFO] Start server with: cd pokemon-showdown && node pokemon-showdown start --no-security")
            return None  # Not a failure, just server not running
        raise
    except Exception as e:
        print(f"  [FAIL] Reset error: {e}")
        return False


def test_environment_step():
    """Test environment step (requires server)."""
    print("\n" + "=" * 60)
    print("Testing environment step (requires server)...")
    print("=" * 60)

    from src.envs import VGCEnv

    try:
        env = VGCEnv()
        print("  [INFO] Resetting environment...")
        obs, info = env.reset()

        # Get a valid action from the mask (shape: 214 = 107 + 107)
        mask = info["action_mask"]
        mask0 = mask[:107]  # Valid actions for position 0
        mask1 = mask[107:]  # Valid actions for position 1

        valid0 = np.where(mask0)[0]
        valid1 = np.where(mask1)[0]

        if len(valid0) == 0 or len(valid1) == 0:
            print("  [WARN] No valid actions found!")
            env.close()
            return False

        # Sample one valid action per position
        a0 = valid0[0]
        a1 = valid1[0]
        action = np.array([a0, a1])

        print(f"  [INFO] Taking action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  [OK] Step successful!")
        print(f"  [OK] Reward: {reward:.4f}")
        print(f"  [OK] Terminated: {terminated}")
        print(f"  [OK] Turn: {info.get('turn', 'N/A')}")

        # Take a few more steps
        steps = 0
        while not terminated and not truncated and steps < 5:
            if "action_mask" in info:
                mask = info["action_mask"]
                mask0 = mask[:107]
                mask1 = mask[107:]
                valid0 = np.where(mask0)[0]
                valid1 = np.where(mask1)[0]
                if len(valid0) > 0 and len(valid1) > 0:
                    a0 = np.random.choice(valid0)
                    a1 = np.random.choice(valid1)
                    action = np.array([a0, a1])
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                else:
                    break
            else:
                break

        print(f"  [OK] Completed {steps + 1} steps")

        env.close()
        return True

    except RuntimeError as e:
        if "server" in str(e).lower() or "timeout" in str(e).lower():
            print(f"  [SKIP] Server not running")
            return None
        raise
    except Exception as e:
        print(f"  [FAIL] Step error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VGC RL Setup Test Suite")
    print("=" * 60)

    results = {}

    # Basic tests (no server required)
    results["imports"] = test_imports()
    results["team_loading"] = test_team_loading()
    results["action_decoding"] = test_action_decoding()
    results["env_creation"] = test_environment_creation()

    # Server-dependent tests
    results["env_reset"] = test_environment_reset()
    if results["env_reset"]:
        results["env_step"] = test_environment_step()
    else:
        results["env_step"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {status} {test}")

    # Overall result
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\n{len(failures)} test(s) failed: {failures}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())

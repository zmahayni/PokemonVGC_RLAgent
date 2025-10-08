import numpy as np
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv

DEFAULT_RESERVED_INDEX = 0

def map_orders_to_indices(valid_orders, battle):
    indices = []
    for order in valid_orders:
        idx_np = SinglesEnv.order_to_action(
            order=order, battle=battle, fake=True, strict=False
        )
        idx = int(idx_np)
        indices.append(idx)
    return indices


def build_mask_from_battle(battle, act_size):
    """
    Build a boolean mask (np.ndarray of dtype=bool, shape [act_size]) where True means
    the action index is legal at this timestep.

    Fallback: if no valid_orders are present (e.g., between episodes), return all True.
    """
    # Default: mask off everything
    mask = np.zeros(act_size, dtype=bool)

    # Between episodes or not ready: allow all to avoid crashes
    if battle is None or battle.valid_orders is None or len(battle.valid_orders) == 0:
        return np.ones(act_size, dtype=bool)

    # Handle default-only turn: map '/choose default' to our reserved discrete index
    vo = battle.valid_orders
    if isinstance(vo, (list, tuple)) and len(vo) == 1 and str(vo[0]).strip() == "/choose default":
        mask[:] = False
        if 0 <= DEFAULT_RESERVED_INDEX < act_size:
            mask[DEFAULT_RESERVED_INDEX] = True
        return mask

    # General case: map valid orders to indices 0..25
    indices = map_orders_to_indices(valid_orders=battle.valid_orders, battle=battle)
    for idx in indices:
        if 0 <= idx < act_size:
            mask[idx] = True

    # Safety: ensure at least one action is allowed
    if not mask.any():
        mask[:] = True

    return mask


def mask_fn(env):
    base = env
    while not isinstance(base, SingleAgentWrapper) and hasattr(base, "env"):
        base = base.env

    poke_env = base.env
    act_size = base.action_space.n
    battle = getattr(poke_env, "battle1", None)
    return build_mask_from_battle(battle=battle, act_size=act_size)


class DebugSingleAgentWrapper(SingleAgentWrapper):
    def _debug_log(self, action):
        # Underlying poke-env (Singles_Env_v1) is self.env
        poke_env = self.env
        act_size = self.action_space.n
        battle = getattr(poke_env, "battle1", None)

        # Build current mask for visibility
        mask = build_mask_from_battle(battle, act_size)
        allowed = [i for i, ok in enumerate(mask) if ok]
        tag = (
            getattr(battle, "battle_tag", "<no battle>")
            if battle is not None
            else "<no battle>"
        )
        turn = getattr(battle, "turn", -1) if battle is not None else -1
        orders = None
        if battle is not None and getattr(battle, "valid_orders", None) is not None:
            try:
                orders = [str(o) for o in battle.valid_orders]
            except Exception:
                orders = None
        print(
            f"[DEBUG] {tag} turn {turn} action={action} allowed={allowed} valid_orders={orders}"
        )

    def step(self, action):
        self._debug_log(action)

        # Translate reserved default index to -2 only if default is the sole valid order
        poke_env = self.env
        battle = getattr(poke_env, "battle1", None)
        vo = None if battle is None else getattr(battle, "valid_orders", None)
        if (
            vo is not None
            and isinstance(vo, (list, tuple))
            and len(vo) == 1
            and str(vo[0]).strip() == "/choose default"
            and int(action) == int(DEFAULT_RESERVED_INDEX)
        ):
            action = np.int64(-2)

        return super().step(action)

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

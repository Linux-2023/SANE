"""Training-time action-prefix conditioning for pi0.5.

The objective matches the RSSR RTC training branch at 25eed0cc / lerobot-pi0
b500ef9d: clean prefixes with per-token t=0 and loss only on in-episode,
non-prefix positions and real action dimensions.
"""

import jax
import jax.numpy as jnp


def training_inputs(rng, actions, noise, time, *, max_prefix_frac: float, prefix_drop_p: float):
    """Return noisy actions, per-position times, and the trained suffix mask."""
    batch, horizon, _ = actions.shape
    length_rng, drop_rng = jax.random.split(rng)
    max_prefix = max(0, min(horizon - 1, int(horizon * max_prefix_frac)))
    lengths = jax.random.randint(length_rng, (batch,), 0, max_prefix + 1)
    drop = jax.random.uniform(drop_rng, (batch,)) < prefix_drop_p
    lengths = jnp.where(drop, 0, lengths)
    suffix_mask = jnp.arange(horizon)[None, :] >= lengths[:, None]
    token_time = jnp.where(suffix_mask, time[:, None], 0.0)
    x_t = token_time[..., None] * noise + (1.0 - token_time[..., None]) * actions
    return x_t, token_time, suffix_mask


def masked_flow_loss(velocity, target, suffix_mask, action_is_pad, *, action_dim: int):
    """Return per-position losses whose mean equals the valid scalar RTC loss."""
    valid = jnp.logical_and(suffix_mask, jnp.logical_not(action_is_pad))
    squared_error = jnp.square(velocity[..., :action_dim] - target[..., :action_dim])
    per_position = jnp.mean(squared_error, axis=-1)
    # The existing trainer reduces the returned BxH array with a global mean.
    # Scale so that this is the mean over valid (non-prefix, in-episode) positions.
    return jnp.where(valid, per_position, 0.0) * (valid.size / jnp.maximum(jnp.sum(valid), 1))

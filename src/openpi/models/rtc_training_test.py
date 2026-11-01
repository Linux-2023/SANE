import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import rtc_training


def test_rtc_inputs_use_clean_prefix_and_per_token_time():
    actions = jnp.full((4096, 50, 3), 2.0)
    noise = jnp.full_like(actions, 6.0)
    time = jnp.full((4096,), 0.25)
    x_t, token_time, suffix = jax.jit(
        lambda key: rtc_training.training_inputs(
            key, actions, noise, time, max_prefix_frac=0.3, prefix_drop_p=0.25
        )
    )(jax.random.key(27))
    lengths = np.sum(~np.asarray(suffix), axis=1)
    assert lengths.min() == 0
    assert lengths.max() == 15
    assert 0.26 < np.mean(lengths == 0) < 0.34
    np.testing.assert_allclose(token_time, np.where(suffix, 0.25, 0.0))
    np.testing.assert_allclose(x_t, np.broadcast_to(np.where(np.asarray(suffix)[..., None], 3.0, 2.0), x_t.shape))

    x_t, token_time, suffix = rtc_training.training_inputs(
        jax.random.key(28), actions[:2], noise[:2], time[:2], max_prefix_frac=0.3, prefix_drop_p=1.0
    )
    assert np.all(suffix)
    np.testing.assert_allclose(token_time, 0.25)
    np.testing.assert_allclose(x_t, 3.0)


def test_rtc_loss_excludes_prefix_episode_tail_and_padded_dimensions():
    suffix = jnp.array([[False, True, True, True], [True, True, False, True]])
    is_pad = jnp.array([[False, False, False, True], [False, False, True, True]])
    velocity = jnp.ones((2, 4, 3)).at[..., 2].set(1000.0)
    target = jnp.zeros_like(velocity)

    def loss(v):
        return jnp.mean(rtc_training.masked_flow_loss(v, target, suffix, is_pad, action_dim=2))

    np.testing.assert_allclose(jax.jit(loss)(velocity), 1.0)
    grad = np.asarray(jax.grad(loss)(velocity))
    expected = np.zeros((2, 4, 3), dtype=np.float32)
    expected[np.asarray(suffix & ~is_pad), :2] = 0.25
    np.testing.assert_allclose(grad, expected)


def test_adarms_supports_independent_time_for_each_action():
    norm = gemma.RMSNorm()
    x = jnp.arange(1, 1 + 2 * 4 * 8, dtype=jnp.float32).reshape(2, 4, 8)
    shared = jnp.ones((2, 8), dtype=jnp.float32)
    params = norm.init(jax.random.key(3), x, shared)
    params["params"]["Dense_0"]["kernel"] = jnp.arange(8 * 24, dtype=jnp.float32).reshape(8, 24) / 1000.0
    repeated = jnp.broadcast_to(shared[:, None, :], (2, 4, 8))
    old_output, old_gate = norm.apply(params, x, shared)
    new_output, new_gate = norm.apply(params, x, repeated)
    np.testing.assert_allclose(old_output, new_output, rtol=1e-6)
    np.testing.assert_allclose(jnp.broadcast_to(old_gate, new_gate.shape), new_gate, rtol=1e-6)
    changed = repeated.at[0, 1].set(2.0)
    changed_output, _ = norm.apply(params, x, changed)
    assert not np.allclose(changed_output[0, 1], new_output[0, 1])
    np.testing.assert_allclose(changed_output[0, 0], new_output[0, 0])
    np.testing.assert_allclose(changed_output[1], new_output[1])

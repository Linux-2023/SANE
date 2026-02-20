"""Gaussian Process utilities for structured noise sampling in Flow Matching.

Provides three kernel functions (SE, OU, PE) and GP noise sampling that
introduces temporal correlations across the action horizon, replacing the
default i.i.d. Gaussian noise.

Designed for action_horizon=50, with execution_horizon=15 in mind.
Recommended length_scale values: SE=5.0, OU=3.0, PE=2.0
"""

from typing import Literal

import jax
import jax.numpy as jnp

# Type alias for supported kernel names
KernelType = Literal["SE", "OU", "PE"]


def squared_exponential_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Squared Exponential (SE) kernel: infinitely smooth samples.

    K_SE(τ, τ') = exp(-d² / (2 * ℓ²))

    Suitable for smooth time series. Stronger temporal correlations over longer
    horizons compared to OU. Recommended length_scale=5.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls correlation range. Higher = smoother.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-d**2 / (2.0 * length_scale**2))


def ornstein_uhlenbeck_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Ornstein-Uhlenbeck (OU) kernel: related to Brownian motion.

    K_OU(τ, τ') = exp(-|d| / ℓ)

    Produces rougher, more jagged samples. Decays faster than SE.
    Recommended length_scale=3.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls correlation range. Higher = longer memory.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-jnp.abs(d) / length_scale)


def periodic_kernel(
    tau: jax.Array,
    tau_prime: jax.Array,
    length_scale: float,
) -> jax.Array:
    """Periodic (PE) kernel: captures repeating patterns.

    K_PE(τ, τ') = exp(-2 * sin²(d) / ℓ²)

    Suitable for time series with periodic behavior (e.g., cyclic motions).
    Recommended length_scale=2.0 for horizon=50.

    Args:
        tau: First time points, shape (...,)
        tau_prime: Second time points, shape (...,)
        length_scale: ℓ, controls smoothness of periodicity.

    Returns:
        Kernel matrix values, same shape as inputs broadcast together.
    """
    d = tau - tau_prime
    return jnp.exp(-2.0 * jnp.sin(d) ** 2 / length_scale**2)


def build_gp_covariance(
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> jax.Array:
    """Build the GP covariance matrix for a given action horizon.

    Time points are uniformly spaced in [0, action_horizon - 1].
    The resulting matrix K has shape (action_horizon, action_horizon).

    Args:
        action_horizon: Number of time steps (e.g., 50).
        kernel_type: One of "SE", "OU", "PE".
        length_scale: Kernel length scale parameter ℓ.
        jitter: Small diagonal term for numerical stability.

    Returns:
        Covariance matrix K of shape (action_horizon, action_horizon).
    """
    tau = jnp.arange(action_horizon, dtype=jnp.float32)
    # Build pairwise difference matrix via broadcasting
    tau_i = tau[:, None]   # (T, 1)
    tau_j = tau[None, :]   # (1, T)

    if kernel_type == "SE":
        K = squared_exponential_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "OU":
        K = ornstein_uhlenbeck_kernel(tau_i, tau_j, length_scale)
    elif kernel_type == "PE":
        K = periodic_kernel(tau_i, tau_j, length_scale)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r}. Choose from 'SE', 'OU', 'PE'.")

    # Add jitter for numerical stability
    K = K + jitter * jnp.eye(action_horizon)
    return K


def sample_gp_noise(
    rng: jax.Array,
    shape: tuple[int, ...],
    action_horizon: int,
    kernel_type: KernelType = "SE",
    length_scale: float = 5.0,
    jitter: float = 1e-5,
) -> jax.Array:
    """Sample temporally-correlated noise using a Gaussian Process prior.

    Replaces i.i.d. `jax.random.normal(...)` with GP-structured noise.
    The temporal correlation is applied along the action_horizon dimension.
    Each action dimension is sampled independently from the same GP.

    Noise is sampled via: z = L @ eps, where L = cholesky(K), eps ~ N(0,I)

    Args:
        rng: JAX random key.
        shape: Full noise shape (batch, action_horizon, action_dim).
        action_horizon: Number of timesteps; must match shape[-2].
        kernel_type: "SE" (smooth), "OU" (rough), or "PE" (periodic).
        length_scale: Kernel length scale ℓ. Recommended values:
            SE=5.0, OU=3.0, PE=2.0 for action_horizon=50, exec_horizon=15.
        jitter: Small diagonal term for numerical stability in Cholesky.

    Returns:
        GP-structured noise of shape `shape`, with temporal correlations
        along the action_horizon axis.
    """
    assert shape[-2] == action_horizon, (
        f"shape[-2]={shape[-2]} must equal action_horizon={action_horizon}"
    )

    # Build covariance matrix K: (action_horizon, action_horizon)
    K = build_gp_covariance(action_horizon, kernel_type, length_scale, jitter)

    # Cholesky decomposition: K = L L^T, L lower-triangular
    # shape: (action_horizon, action_horizon)
    L = jnp.linalg.cholesky(K)

    # Sample standard Gaussian: shape (batch, action_horizon, action_dim)
    eps = jax.random.normal(rng, shape)

    # Apply temporal correlation: z = L @ eps along the action_horizon axis
    # eps: (..., T, D) -> einsum over T dimension
    # L: (T, T), eps: (..., T, D) -> noise: (..., T, D)
    noise = jnp.einsum("ij,...jd->...id", L, eps)

    return noise

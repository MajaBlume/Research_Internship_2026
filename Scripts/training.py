"""
ude.training
============

Training utilities for Universal Differential Equations.

Functions
---------
dataloader    — infinite random mini-batch generator
train_model   — two-phase curriculum training loop (AdaBelief → convergence)
multi_start   — run K independent trials, return the best solution

Design decisions
----------------
**Two-phase curriculum**: biological ODE training benefits from first fitting
on short time windows (where the ODE is well-behaved and the loss landscape is
smoother) before extending to the full trajectory.  ``steps_strategy`` and
``length_strategy`` control this curriculum.

**AdaBelief optimiser**: a variant of Adam with improved convergence on
non-convex landscapes, well-suited to the mixed mechanistic/neural parameter
space.  Learning rate 1e-3 is a robust default; reduce to 1e-4 for the
refinement phase if loss stagnates.

**Gradient clipping**: applied by default (max-norm 10.0) to prevent
instabilities during the early training phase.

**Multi-start**: the UDE loss landscape has many local minima.  Running K
independent optimisations with different random seeds and selecting the best
solution significantly improves the probability of finding a good solution.
Runs are executed sequentially by default but can be parallelised with
``Threads.@threads`` or by launching separate processes.
"""

from __future__ import annotations

import time
from typing import Callable, Generator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

Array = "jax.Array"


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def dataloader(
    arrays: tuple[Array, ...],
    batch_size: int,
    *,
    key: Array,
) -> Generator[tuple[Array, ...], None, None]:
    """
    Infinite mini-batch generator that shuffles the dataset every epoch.

    Yields tuples of arrays, each of length ``batch_size`` along the first
    axis.  Shuffling is done with a JAX PRNG key, so the sequence of batches
    is fully reproducible given the same ``key``.

    Parameters
    ----------
    arrays : tuple of Arrays
        Dataset arrays.  All must have the same first-axis size
        (number of trajectories / samples).
    batch_size : int
        Number of samples per batch.
    key : Array
        PRNG key for shuffling.

    Yields
    ------
    tuple of Arrays
        One batch per ``yield``.

    Examples
    --------
    >>> loader = dataloader((ys, y0), batch_size=4, key=jax.random.PRNGKey(0))
    >>> yi, y0i = next(loader)
    """
    dataset_size = arrays[0].shape[0]
    indices = jnp.arange(dataset_size)

    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, indices)
        for start in range(0, dataset_size - batch_size + 1, batch_size):
            batch_idx = perm[start : start + batch_size]
            yield tuple(arr[batch_idx] for arr in arrays)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_model(
    problem: eqx.Module,
    data: Array,
    y0: Array,
    ts_train: Array,
    *,
    loader_key: Array,
    steps_strategy: tuple[int, ...] = (500, 500),
    length_strategy: tuple[float, ...] = (0.5, 1.0),
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    grad_clip: float = 10.0,
    print_every: int = 100,
    scale: Array = None,
    filter_spec = None,
    verbose: bool = True,
) -> eqx.Module:
    """
    Two-phase curriculum training for an ODE problem.

    Training proceeds in phases defined by ``steps_strategy`` and
    ``length_strategy``.  In each phase, the model is trained on an initial
    segment of the trajectory (controlled by ``length_strategy``).  This
    curriculum eases the model into fitting the full trajectory and avoids
    gradient explosions in early training.

    Parameters
    ----------
    problem : eqx.Module
        An ``OdeProblem`` wrapping the model to train.
    data : Array, shape (n_traj, T, n_obs)
        Target observable trajectories (typically noisy).
    y0 : Array, shape (n_traj, n_states)
        Initial conditions (full state, not just observables).
    ts_train : Array, shape (T,)
        Full training time grid.
    loader_key : Array
        PRNG key for the dataloader shuffling.
    steps_strategy : tuple of int, optional
        Number of gradient steps in each phase.  Default (500, 500).
    length_strategy : tuple of float, optional
        Fraction of the trajectory to use in each phase.  Default (0.5, 1.0)
        means phase 1 uses the first 50% of timepoints, phase 2 uses all.
    batch_size : int, optional
        Mini-batch size (number of trajectories per gradient step).  Default 1.
    learning_rate : float, optional
        AdaBelief learning rate.  Default 1e-3.
    grad_clip : float, optional
        Global gradient clipping threshold (max-norm).  Default 10.0.
    print_every : int, optional
        Print loss every this many steps.  Default 100.
    scale : array, shape (n_states,)
    verbose : bool, optional
        If False, suppress all printed output.  Default True.

    Returns
    -------
    eqx.Module
        Trained model (same type as input ``problem``).

    Examples
    --------
    >>> trained = train_model(
    ...     problem, data_train, y0_full, ts_train,
    ...     loader_key=jax.random.PRNGKey(1),
    ...     steps_strategy=(300, 700),
    ...     length_strategy=(0.4, 1.0),
    ...     learning_rate=1e-3,
    ... )
    """
    _, length_size, _ = data.shape

    # ---- Optimiser: AdaBelief with optional gradient clipping -----------
    optim = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adabelief(learning_rate),
    )
    

    # ---- Loss and gradient functions ------------------------------------
    @eqx.filter_value_and_grad
    def grad_loss(model, ti: Array, yi: Array, y0_i: Array) -> Array:
        """MSE loss over a mini-batch of trajectories."""
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, y0_i)
        return jnp.mean((yi - y_pred) ** 2)
    
    @eqx.filter_value_and_grad
    def grad_loss_scaled(model, ti: Array, yi: Array, y0_i: Array, scale: Array,) -> Array:
        """MSE loss over a mini-batch of trajectories."""
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, y0_i)
        return jnp.mean(((yi - y_pred)/scale) ** 2)
    
    @eqx.filter_value_and_grad
    def grad_loss_scaled_fixed(diff_model,static_model, ti: Array, yi: Array, y0_i: Array, scale: Array,) -> Array:
        """MSE loss over a mini-batch of trajectories."""
        model = eqx.combine(diff_model, static_model)
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, y0_i)
        return jnp.mean(((yi - y_pred)/scale) ** 2)


    @eqx.filter_jit
    def make_step(
        ti: Array,
        yi: Array,
        y0_i: Array,
        model: eqx.Module,
        opt_state,
        scale: Array = None,
        filter_spec = None
    ):
        if scale == None:
            loss, grads = grad_loss(model, ti, yi, y0_i)
        
        else:
            if filter_spec == None:
                loss, grads = grad_loss_scaled(model, ti, yi, y0_i, scale)
            else:
                diff_model, static_model = eqx.partition(model, filter_spec)
                loss, grads = grad_loss_scaled_fixed(diff_model, static_model, ti, yi, y0_i, scale)

        updates, new_opt_state = optim.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return loss, new_model, new_opt_state

    # ---- Phase loop -----------------------------------------------------
    for phase_idx, (steps, length) in enumerate(
        zip(steps_strategy, length_strategy, strict=True)
    ):
        opt_state = optim.init(eqx.filter(problem, eqx.is_inexact_array))
        n_pts = int(length_size * length)
        _ts = ts_train[:n_pts]
        _ys = data[:, :n_pts, :]

        if verbose:
            print(f"\n── Phase {phase_idx + 1}/{len(steps_strategy)} "
                  f"({steps} steps, t ∈ [{_ts[0]:.2f}, {_ts[-1]:.2f}]) ──")

        for step, (yi, y0_i) in zip(
            range(steps),
            dataloader((_ys, y0), batch_size, key=loader_key),
            strict=False,
        ):
            t0 = time.perf_counter()
            loss, problem, opt_state = make_step(_ts, yi, y0_i, problem, opt_state, scale, filter_spec)
            elapsed = time.perf_counter() - t0

            if verbose and ((step % print_every == 0) or step == steps - 1):
                print(f"  Step {step:>5d} | Loss {float(loss):.6f} | "
                      f"Time {elapsed:.3f}s")

    return problem

    
    
"""
ude.solver
==========

Wraps any model right-hand side into a differentiable ODE problem using
`diffrax <https://docs.kidger.site/diffrax/>`_.

The ``OdeProblem`` class handles:

- Numerical integration with an adaptive Runge-Kutta solver (Tsit5)
- PID step-size control for stiff and near-stiff systems
- Selecting observable species from the full state trajectory
- Batching across multiple initial conditions via ``jax.vmap``

Design notes
------------
All diffrax parameters are exposed as class attributes so they can be
overridden when needed (e.g. tighter tolerances for publication-quality
figures, looser tolerances for rapid prototyping).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import diffrax

Array = "jax.Array"


class OdeProblem(eqx.Module):
    """
    Differentiable ODE initial-value problem.

    Wraps any callable ``rhs(t, y, args) -> dy/dt`` into an object that,
    when called with a time grid and initial condition, returns the predicted
    observable time series.

    Parameters
    ----------
    rhs : eqx.Module
        The ODE right-hand side.  Must be a JAX-compatible callable with
        signature ``(t: float, y: Array, args: Any) -> Array``.
    full_state : bool, default True
        If ``True``, return only the observable species defined in
        ``OBS_IDX`` (typically [N2, A3]).
        If ``False``, return the full state trajectory.
    rtol : float, default 1e-3
        Relative tolerance for the adaptive step-size controller.
    atol : float, default 1e-6
        Absolute tolerance for the adaptive step-size controller.
    max_steps : int, default 16_384
        Maximum number of ODE solver steps.  Increase if the solver
        terminates early for stiff or long-horizon problems.
    solver : solver

    Examples
    --------
    >>> from ude import ODEcore, OdeProblem, RUOFF_PARAMS_CORE, Y0_RUOFF
    >>> import jax.numpy as jnp
    >>> ts = jnp.linspace(0.0, 5.0, 501)
    >>> prob = OdeProblem(ODEcore(RUOFF_PARAMS_CORE), full_state=True)
    >>> pred = prob(ts, Y0_RUOFF)   # shape (501, 2) — [N2, A3]

    Batched prediction over multiple initial conditions
    ---------------------------------------------------
    >>> import jax
    >>> y0_batch = jnp.stack([Y0_RUOFF, Y0_RUOFF * 1.1])   # (2, 7)
    >>> preds = jax.vmap(prob, in_axes=(None, 0))(ts, y0_batch)  # (2, 501, 2)

    Notes
    -----
    The solver uses ``diffrax.Tsit5``, a 5th-order explicit Runge-Kutta
    method well-suited to non-stiff biological oscillators.  For stiff systems
    (e.g. with very fast and slow timescales), replace with
    ``diffrax.Kvaerno5`` or ``diffrax.Rodas5P``.
    """

    rhs: eqx.Module
    full_state: bool
    rtol: float
    atol: float
    max_steps: int

    def __init__(
        self,
        rhs: eqx.Module,
        full_state: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 16_384,
    ) -> None:
        self.rhs = rhs
        self.full_state = full_state
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def __call__(self, ts: "jax.Array", y0: "jax.Array") -> "jax.Array":
        """
        Solve the IVP and return the observable time series.

        Parameters
        ----------
        ts : Array, shape (T,)
            Time grid at which to save the solution.  Must be sorted and
            the solver integrates from ``ts[0]`` to ``ts[-1]``.
        y0 : Array, shape (n_states,)
            Initial condition.

        Returns
        -------
        Array, shape (T, n_obs) or (T, n_states)
            Predicted concentrations at each time point.
            When ``full_state=True`` -> shape (T, len(OBS_IDX)).
            When ``full_state=False`` -> shape (T, n_states).
        """
        # solution = diffrax.diffeqsolve(
        #     diffrax.ODETerm(self.rhs),
        #     diffrax.Kvaerno5(),  # stiff solver; replace with Tsit5() for non-stiff systems
        #     t0=ts[0],
        #     t1=ts[-1],
        #     dt0=ts[1] - ts[0],
        #     y0=y0,
        #     stepsize_controller=diffrax.PIDController(
        #         rtol=self.rtol,
        #         atol=self.atol,
        #     ),
        #     saveat=diffrax.SaveAt(ts=ts),
        #     max_steps=self.max_steps,
        # )
        # ys = solution.ys  # shape: (T, n_states)
        # return ys                   # (T, n_states)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.rhs),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=1.0,
            y0=y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol,
                atol=self.atol,
            ),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=300_000,
        )
        ys = solution.ys  # shape: (T, n_states)
        return ys                   # (T, n_states)


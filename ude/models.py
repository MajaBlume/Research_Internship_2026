from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = jax.Array
ConstraintMode = Literal["none", "x", "tanh"]
OperatorMode = Literal["add", "mul", "pow"]


# ===========================================================================
# NDE — Neural Differential Equation
# ===========================================================================

class NDE(eqx.Module):
    """
    A fully data-driven model: a multi-layer perceptron (MLP) approximates
    the full vector field.

    The MLP maps the current state ``y ∈ ℝ^in_size`` to a time-derivative
    vector ``dy/dt ∈ ℝ^out_size``.  When used inside a UDE, ``out_size``
    should match the state dimension of the mechanistic ODE so that the
    outputs can be added element-wise.

    Parameters
    ----------
    in_size : int
        Dimension of the state vector (input to the MLP).
    out_size : int
        Dimension of the output (should equal state dimension for a UDE).
    width_size : int
        Number of neurons per hidden layer.
    depth : int
        Number of hidden layers (not counting input/output layers).
    key : jax.Array
        PRNG key for weight initialisation.

    Examples
    --------
    >>> nde = NDE(in_size=7, out_size=7, width_size=32, depth=2, key=jax.random.PRNGKey(0))
    >>> nde(t=0.0, y=jnp.ones(7), args=None)   # returns shape (7,)
    """

    out_scale: Array
    mlp: eqx.nn.MLP

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        *,
        key: Array,
    ) -> None:
        # Learnable global output scale (initialised at 1)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.swish,       # smooth & good for ODEs
            key=key,
        )

    def __call__(self, t: float, y: Array, args) -> Array:
        """Evaluate the neural vector field at state ``y``."""
        return self.out_scale * self.mlp(y)


# ===========================================================================
# ODEs The Construction of a community model of S. elongatus, 
#   U. maydis and  (S. cerevisiae)
# ===========================================================================

class Coculture_Final_Model (eqx.Module):
    
    """
        State Vector ( Components)
                --------------------------------
                Se: S. elongatus
                Um: U. maydis
                Suc: Sucrose
                N: Nitrogen
                P1: Parameter1
                Met1: Metabolite1
                Met2: Metabolite2
            
            Parameter vectors
                ---------------------------------
                    
                    parameters_Se: Array
                    -----------------------------
                    u_max_Se, k_L_Se, e_Se,  k_N_Se, Y_Se_N, k_death_Se,
                    beta_Suc_Se, k_Met1_Se, beta_Met2_Se, k_P1_Met2_Se, K_I_pH_Suc_exp
                     
                    parameters_Um: Array
                    -----------------------------
                    u_max_Um, k_Suc_Um, Y_Um_Suc, k_N_Um,Y_Um_N, k_death_Um,
                    e_Um, beta_Met1_Um, K_I_Met1_Um, K_I_Met2_Um, k_Met2_Um
                    
                    parameters_abiotic: Array
                    -----------------------------
                    L_0, Suc_Feed
                
            
            weights: Array
    """
    parameters_Se: Array
    parameters_Um: Array
    parameters_abiotic: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        
        # parameters and variables:
        # ----------------------------------------------------
        Se,Um,Suc,N,P1,Met1,Met2, = variables
        u_max_Se, k_L_Se, e_Se,  k_N_Se, Y_Se_N, k_death_Se, alpha_Suc_Se, k_Met1_Se, Y_Se_Met1, beta_Met2_Se = self.parameters_Se
        u_max_Um,k_Suc_Um, Y_Um_Suc, k_N_Um,Y_Um_N, k_death_Um, e_Um, beta_Met1_Um, K_I_Met1_Um, K_I_Met2_Um = self.parameters_Um
        L_0, e_medium, = self.parameters_abiotic
    
        # derived
        L = L_0 * jnp.exp(-Se*e_Se - Um*e_Um - e_medium)
        
        # growth of Se
        u_Se    = u_max_Se * (L / (k_L_Se + L)) * (N / (k_N_Se + N)) * (1 + (Met1 / (k_Met1_Se + Met1)))
        Met1_imp_Se = Se * u_Se * (Met1 / (k_Met1_Se + Met1)) / Y_Se_Met1
        Suc_exp_Se  = Se * alpha_Suc_Se * L
        
        # growth of Um
        u_Um    = u_max_Um * (Suc / (k_Suc_Um + Suc)) * (N / (k_N_Um + N)) * (K_I_Met2_Um / (K_I_Met2_Um + Met2))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        Met1_exp_Um = Um * u_Um * beta_Met1_Um * (K_I_Met1_Um / (Met1 + K_I_Met1_Um))
        
        # change of variables
        #---------------------------------------------------------
        # microbes
        dSedt       = Se * (u_Se - k_death_Se)
        dUmdt       = Um * (u_Um - k_death_Um)

        
        # paramters, metabolites, etc. (ressources)
        dSucdt      = Suc_exp_Se - Suc_imp_Um
        dNdt        = - u_Se * Se * 1/Y_Se_N - u_Um * Um * 1/Y_Um_N
        dP1dt       = 0.0
        dMet1dt     = Met1_exp_Um - Met1_imp_Se
        dMet2dt     = 0.0

        return jnp.array([dSedt,dUmdt,dSucdt,dNdt,dP1dt,dMet1dt,dMet2dt])
    
    
class Coculture_no_Met1 (eqx.Module):
    
    """
        State Vector ( Components)
                --------------------------------
                Se: S. elongatus
                Um: U. maydis
                Suc: Sucrose
                N: Nitrogen
                P1: Parameter1
                Met1: Metabolite1
                Met2: Metabolite2
            
            Parameter vectors
                ---------------------------------
                    
                    parameters_Se: Array
                    -----------------------------
                    u_max_Se, k_L_Se, e_Se,  k_N_Se, Y_Se_N, k_death_Se,
                    beta_Suc_Se, k_Met1_Se, beta_Met2_Se, k_P1_Met2_Se, K_I_pH_Suc_exp
                     
                    parameters_Um: Array
                    -----------------------------
                    u_max_Um, k_Suc_Um, Y_Um_Suc, k_N_Um,Y_Um_N, k_death_Um,
                    e_Um, beta_Met1_Um, K_I_Met1_Um, K_I_Met2_Um, k_Met2_Um
                    
                    parameters_abiotic: Array
                    -----------------------------
                    L_0, Suc_Feed
                
            
            weights: Array
    """
    parameters_Se: Array
    parameters_Um: Array
    parameters_abiotic: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        
        # parameters and variables:
        # ----------------------------------------------------
        Se,Um,Suc,N,P1,Met1,Met2, = variables
        u_max_Se, k_L_Se, e_Se,  k_N_Se, Y_Se_N, k_death_Se, alpha_Suc_Se, k_Met1_Se, Y_Se_Met1, beta_Met2_Se = self.parameters_Se
        u_max_Um,k_Suc_Um, Y_Um_Suc, k_N_Um,Y_Um_N, k_death_Um, e_Um, beta_Met1_Um, K_I_Met1_Um, K_I_Met2_Um = self.parameters_Um
        L_0, e_medium, = self.parameters_abiotic
    
        # derived
        L = L_0 * jnp.exp(-Se*e_Se - Um*e_Um - e_medium)
        
        # growth of Se
        u_Se    = u_max_Se * (L / (k_L_Se + L)) * (N / (k_N_Se + N)) # * (1 + (Met1 / (k_Met1_Se + Met1)))
        #Met1_imp_Se = Se * u_Se * (Met1 / (k_Met1_Se + Met1)) / Y_Se_Met1
        Suc_exp_Se  = Se * alpha_Suc_Se * L
        
        # growth of Um
        u_Um    = u_max_Um * (Suc / (k_Suc_Um + Suc)) * (N / (k_N_Um + N)) * (K_I_Met2_Um / (K_I_Met2_Um + Met2))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        #Met1_exp_Um = Um * u_Um * beta_Met1_Um * (K_I_Met1_Um / (Met1 + K_I_Met1_Um))
        
        # change of variables
        #---------------------------------------------------------
        # microbes
        dSedt       = Se * (u_Se - k_death_Se)
        dUmdt       = Um * (u_Um - k_death_Um)

        
        # paramters, metabolites, etc. (ressources)
        dSucdt      = Suc_exp_Se - Suc_imp_Um
        dNdt        = - u_Se * Se * 1/Y_Se_N - u_Um * Um * 1/Y_Um_N
        dP1dt       = 0.0
        dMet1dt     = 0.0 # Met1_exp_Um - Met1_imp_Se
        dMet2dt     = 0.0

        return jnp.array([dSedt,dUmdt,dSucdt,dNdt,dP1dt,dMet1dt,dMet2dt])


# ===========================================================================
# ODEcore — Mechanistic Skeleton (k5-term intentionally removed)
# ===========================================================================

class ODE_LV_core(eqx.Module):
    """
    For the start to play around different variants of Lodka Volterra were tried out
    """

    weights: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        Pray, Pred = variables
        r1, r2, c2 = self.weights

        dPraydt   = Pray*r1
        dPreddt   = -Pred *r2+ Pred*Pray*c2

        return jnp.array([dPraydt, dPreddt])


# ===========================================================================
# UDE: Universal Differential Equation (hybrid)
# ===========================================================================

class UDE_Coculture(eqx.Module):
    """
    In general we compute our UDE as follows UDE = ODE + constraint(NDE)
    However for the first two nodes or the species we comppute: ODE + s(t) * constraint(NDE)
    s(t) is the species abundance at the timepoint. This makes sure that S. c. can't be produced 
    if it was absent in the beginning.
    """

    ode: eqx.Module
    nde: eqx.Module
    lambda_: float

    def __init__(
        self,
        ode: eqx.Module,
        nde: eqx.Module,
        lambda_: float = 1.0,
    ) -> None:
        
        self.ode = ode
        self.nde = nde
        self.lambda_ = lambda_


    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, t: float, y: Array, args) -> Array:
        """Evaluate the UDE vector field at time ``t`` and state ``y``."""
        ode_out = self.ode(t, y, args)
        nde_out = self.nde(t, y, args)
        
        species_scale = jnp.ones_like(y)
        species_scale = species_scale.at[0:2].set(y[0:2])
        return ode_out + self.lambda_ * species_scale * nde_out
    
class NSM(eqx.Module):
    
    """
    Reconstruction of an easy form of NSM
    
    """
    nde: eqx.Module
    parameters: Array

    def __init__(
        self,
        nde: eqx.Module,
        parameters: Array,  
    ) -> None:
        
        self.nde = nde
        self.parameters = parameters

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, t: float, y: Array, args) -> Array:
        """Evaluate the UDE vector field at time ``t`` and state ``y``."""
        
        (P_S1_M1, P_S2_M1, C_S1_M1, C_S2_M1, b_M1, 
        P_S1_M2, P_S2_M2, C_S1_M2,C_S2_M2, b_M2,
        P_S1_M3, P_S2_M3, C_S1_M3, C_S2_M3, b_M3, 
        P_S1_M4, P_S2_M4, C_S1_M4, C_S2_M4, b_M4) = self.parameters
        
        def soft10(x: float) -> float:
            """Calcultates the softplus function with base 10."""
            return jnp.log10(1 + jnp.exp(x))
        
        nde_out = self.nde(t, y, args)
        S1 = y[0]
        S2 = y[1]
        M1 = y[2]
        M2 = y[3]
        P1 = y[4]
        M3 = y[5]
        M4 = y[6]
        
        dS1dt = S1 * nde_out[0]
        dS2dt = S2 * nde_out[1]
        dM1dt = jax.nn.softplus(nde_out[2]) * (S1 * soft10(P_S1_M1) + S2 * soft10(P_S2_M1) - M1 * (S1 * soft10(C_S1_M1) + S2 * soft10(C_S2_M1) + soft10(b_M1)))
        dM2dt = jax.nn.softplus(nde_out[3]) * (S1 * soft10(P_S1_M2) + S2 * soft10(P_S2_M2) - M2 * (S1 * soft10(C_S1_M2) + S2 * soft10(C_S2_M2) + soft10(b_M2)))
        dP1dt = 0.0
        dM3dt = jax.nn.softplus(nde_out[5]) * (S1 * soft10(P_S1_M3) + S2 * soft10(P_S2_M3) - M3 * (S1 * soft10(C_S1_M3) + S2 * soft10(C_S2_M3) + soft10(b_M3)))
        dM4dt = jax.nn.softplus(nde_out[6]) * (S1 * soft10(P_S1_M4) + S2 * soft10(P_S2_M4) - M4 * (S1 * soft10(C_S1_M4) + S2 * soft10(C_S2_M4) + soft10(b_M4)))

        return jnp.array([dS1dt,dS2dt,dM1dt,dM2dt,dP1dt,dM3dt,dM4dt])   
    
    
class UDE(eqx.Module):
    
    """
    Universal Differential Equation: mechanistic ODE + neural network.

    The UDE combines a mechanistic ODE skeleton (``ode``) with a neural
    network (``nde``) to approximate unknown or intractable terms.  The
    combination is:

        UDE(t, y) = combine(ODE(t, y),  λ · constraint(ODE(t, y)) · NDE(t, y))

    where ``combine`` is one of:  add, mul, pow.

    This flexible formulation allows:
    - **Additive correction**: ``operator="add"`` (most common) — the NN adds
      a correction term on top of the mechanistic baseline.
    - **Multiplicative scaling**: ``operator="mul"`` — the NN scales the
      mechanistic output (useful for modelling fold-changes).
    - **Power-law**: ``operator="pow"`` — less common, used for allometric
      relationships.

    The ``constraint`` parameter enforces that the NN correction is
    biologically plausible by gating it through a function of the ODE output:
    - ``"none"``  — no constraint (gate = 1)
    - ``"x"``     — gate = ODE output itself (zeros correction where ODE = 0)
    - ``"tanh"``  — gate = tanh(ODE output), bounded in (-1, 1)

    Parameters
    ----------
    ode : eqx.Module
        Mechanistic right-hand side (e.g. ``ODEcore``).
    nde : eqx.Module
        Neural network right-hand side (e.g. ``NDE``).
    operator : {"add", "mul", "pow"}, default "add"
        How to combine ODE and NDE outputs.
    lambda_ : float, default 2.0
        Scaling factor for the neural network contribution.
    constraint : {"none", "x", "tanh"}, default "none"
        Constraint applied to ODE output before multiplying with NDE.

    """

    ode: eqx.Module
    nde: eqx.Module
    operator: str
    lambda_: float
    constraint: str

    def __init__(
        self,
        ode: eqx.Module,
        nde: eqx.Module,
        operator: OperatorMode = "mul",
        lambda_: float = 2.0,
        constraint: ConstraintMode = "none",
    ) -> None:
        if operator not in ("add", "mul", "pow"):
            raise ValueError(f"operator must be 'add', 'mul', or 'pow', got {operator!r}")
        if constraint not in ("none", "x", "tanh"):
            raise ValueError(f"constraint must be 'none', 'x', or 'tanh', got {constraint!r}")
        self.ode = ode
        self.nde = nde
        self.operator = operator
        self.lambda_ = lambda_
        self.constraint = constraint

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gate(self, ode_out: Array) -> Array:
        """Compute the constraint gate from the ODE output."""
        if self.constraint == "none":
            return jnp.ones_like(ode_out)
        if self.constraint == "x":
            return ode_out
        if self.constraint == "tanh":
            return jax.nn.tanh(ode_out)
        # unreachable — validated in __init__
        raise ValueError(self.constraint)

    def _combine(self, ode_out: Array, nde_out: Array) -> Array:
        """Combine ODE and scaled NDE outputs using the chosen operator."""
        scaled_nde = self.lambda_ * self._gate(ode_out) * nde_out
        if self.operator == "add":
            return ode_out + scaled_nde
        if self.operator == "mul":
            return ode_out * scaled_nde
        if self.operator == "pow":
            return ode_out ** scaled_nde
        raise ValueError(self.operator)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, t: float, y: Array, args) -> Array:
        """Evaluate the UDE vector field at time ``t`` and state ``y``."""
        ode_out = self.ode(t, y, args)
        nde_out = self.nde(t, y, args)
        return self._combine(ode_out, nde_out)
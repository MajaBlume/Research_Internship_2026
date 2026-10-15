"""
ude.models
==========

Model classes for the Universal Differential Equation framework.

Four model types are provided, all compatible with diffrax as ODE right-hand
sides and with equinox's pytree/JIT machinery:

    NDE     — Neural Differential Equation (fully data-driven)
    ODE     — Full Ruoff glycolysis model with trainable k5
    ODEcore — Mechanistic skeleton with the k5-term intentionally removed
    UDE     — Hybrid = ODEcore  +  λ · constraint(ODEcore)  ·  NDE

All classes inherit from ``eqx.Module``, making them immutable JAX pytrees
that support ``jax.jit``, ``jax.grad``, and ``jax.vmap`` transparently.

References
----------
Ruoff, P. et al. (2003). The Goodwin model: simulating the effect of
    cycloheximide and heat shock on the circadian clock of Neurospora crassa.
    J. Theor. Biol., 209(1), 29–42.

Philipps, M. et al. (2025). Universal differential equations in systems
    biology. (preprint)
"""

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
#   U. maydis and S. cerevisiae
# ===========================================================================

class Co_Culture (eqx.Module):
    """
    This is the basic model of sucrose cross feeding which only takes into 
    account growth of S. elongatus following monod kinetic dependent on 
    photon availability, a growth coupled export of Sucrose by S. elongatus 
    and the growth of U. maydis following monod kinetic dependent on sucrose
    availability as well as the growth coupled import of sucrose by U. maydis.
    Photon availability is derived by Lambert Beer and absobance coefficients 
    of S. elongatus and U. maydis
    
    State Vector (3 Components)
        --------------------------------
        Se    — S. elongatus
        Suc   — Sucrose
        Um    - U. maydis
    
    Parameter vector (9 components)
        ---------------------------------
        u_max_Se, k_L_Se, L_0, e_Se, k_death_Se, alpha_Suc_Se, u_max_Um, 
        k_Suc_Um, Y_Um_Suc, k_death_Um, e_Um
    
    weights: Array
    u_max_Se: float
    
    """

    def __call__(self, t: float, variables: Array, args) -> Array:
        Se,Suc, Um= variables
        u_max_Se, k_L_Se, L_0, e_Se, k_death_Se, beta_Suc_Se, u_max_Um, k_Suc_Um, Y_Um_Suc, k_death_Um, e_Um = self.weights
        
        # derived photon availability
        L       = L_0*jnp.exp(-Se*e_Se - Um*e_Um)
        
        # specific growth of Se
        u_Se    = u_max_Se * (L /(k_L_Se + L))
        
        # specific growth of Um
        u_Um    = u_max_Um * (Suc /(k_Suc_Um + Suc))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        
        # Sucrose export by Se
        Suc_exp_Se  = Se * L * beta_Suc_Se
       
        # change of variables
        dSedt   = Se * (u_Se - k_death_Se)
        dUmdt   = Um * (u_Um - k_death_Um)
        dSucdt  = Suc_exp_Se - Suc_imp_Um
    
        return jnp.array([dSedt, dSucdt, dUmdt])


class Co_Culture_Nitrogen (eqx.Module):
    """
    Added Nitrogen to Co_culture: 
    - both microbes are following monod kinetics if limited by Nitrogen
    - Nitrogen is only imported by microbes and has to be in the medium to enable growth
    
    State Vector (3 Components)
            --------------------------------
            Se    — S. elongatus
            Suc   — Sucrose
            Um    - U. maydis
            N     - Nitrogen
        
        Parameter vector (9 components)
            ---------------------------------
            u_max_Se, k_L_Se, L_0, e_Se, k_death_Se, alpha_Suc_Se, k_b, u_max_Um, k_Suc_Um,
            Y_Um_Suc, k_death_Um, e_Um, k_N_Se, k_N_Um, Y_Se_N, Y_Um_N
        
        weights: Array
    """

    weights: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        Se,Suc, Um, N, = variables
        u_max_Se, k_L_Se, L_0, e_Se, k_death_Se, beta_Suc_Se, k_b, u_max_Um, k_Suc_Um, Y_Um_Suc, k_death_Um, e_Um, k_N_Se, k_N_Um, Y_Se_N, Y_Um_N = self.weights
        
        # derived
        L       = L_0*jnp.exp(-Se*e_Se - Um*e_Um)
        
        # growth of Se
        u_Se    = u_max_Se * (L /(k_L_Se + L)) * (N/(k_N_Se+N))
        
        # growth of Um
        u_Um    = u_max_Um * (Suc /(k_Suc_Um + Suc)) * (N/(k_N_Um+N))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        
        # Sucrose export by Se
        Suc_exp_Se  = Se * u_Se * alpha_Suc_Se
        metabolic_burdon_sucrose =k_b * L * beta_Suc_Se
       
        # change of variables
        dSedt   = Se * (u_Se - k_death_Se) - metabolic_burdon_sucrose
        dUmdt   = Um * (u_Um - k_death_Um)
        dSucdt  = Suc_exp_Se - Suc_imp_Um
        dNdt    = -dSedt * 1/Y_Se_N - dUmdt * 1/Y_Um_N
    

        return jnp.array([dSedt, dSucdt, dUmdt, dNdt])
    
    
class Community_Nitrogen_Met1_Suc_Feed (eqx.Module):
    
    """
        Added to Co_culture_Nitrogen:
        - S. cerevisiae
        - Met 1
        
        State Vector ( Components)
                --------------------------------
                Se    — S. elongatus
                Suc   — Sucrose
                Um    - U. maydis
                N     - Nitrogen
                Met1  - Metabolite 1
                Sc    - S. cerevisiea
            
            Parameter vector ( components)
                ---------------------------------
               
            
            weights: Array
    """
    
    # constant feed
    weights: Array
    u_max_Se: float

    def __call__(self, t: float, variables: Array, args) -> Array:
        Se,Suc, Um, N, Met1, Sc, = variables
        Suc = jnp.maximum(0,Suc)
        N = jnp.maximum(0,N)
        Met1 =jnp.maximum(0, Met1)
        Sc = jnp.maximum(0,Sc)
        Um =jnp.maximum(0,Um)
        Se =jnp.maximum(0,Se)
        
        
        k_L_Se, L_0, e_Se, k_death_Se, alpha_Suc_Se, k_b,u_max_Um,k_Suc_Um, Y_Um_Suc, k_death_Um, e_Um,k_N_Se, k_N_Um, Y_Se_N,Y_Um_N, k_imp_Met1_Se, k_Met1_Se, beta_Met1_Um, u_max_Sc, k_Suc_Sc, Y_Sc_Suc, k_N_Sc, Y_N_Sc, k_death_Sc, e_Sc, Suc_feed   = self.weights
        # derived
        L       = L_0*jnp.exp(-Se*e_Se - Um*e_Um - Sc*e_Sc)
        
        # growth of Se
        u_Se    = self.u_max_Se * (L /(k_L_Se + L)) * (N/(k_N_Se+N)) 
        Met1_imp_Se = Se*k_imp_Met1_Se*(Met1/(k_Met1_Se+Met1))
        u_Se    = u_Se*(1+(Met1/(k_Met1_Se+Met1)))
        
        #Saccharomyces Cerevisiae
        u_Sc    = u_max_Sc * (Suc /(k_Suc_Sc + Suc)) * (N/(k_N_Sc+N))
        Suc_imp_Sc = u_Sc * Sc * 1/Y_Sc_Suc
        
        # growth of Um
        u_Um    = u_max_Um * (Suc /(k_Suc_Um + Suc)) * (N/(k_N_Um+N))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        Met1_exp_Um= Um * u_Um * beta_Met1_Um
        
        # Sucrose export by Se
        Suc_exp_Se  = Se * L * alpha_Suc_Se
       
        # change of variables
        dSedt       = Se * (u_Se - k_death_Se)
        dUmdt       = Um * (u_Um - k_death_Um)
        dSucdt      = Suc_exp_Se - Suc_imp_Um - Suc_imp_Sc + Suc_feed
        dNdt        = -u_Se*Se * 1/Y_Se_N - u_Um*Um * 1/Y_Um_N - u_Sc*Sc * 1/Y_N_Sc 
        dMet1dt     = Met1_exp_Um - Met1_imp_Se
        dScdt       = Sc * (u_Sc - k_death_Sc)

        return jnp.array([dSedt, dSucdt, dUmdt, dNdt, dMet1dt, dScdt])



class Axenic_Se(eqx.Module):


    weights: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        Se= variables
        u_max_Se, k_L_Se, L_0, e_Se, k_death_Se = self.weights
        
        # derived
        L           = L_0*jnp.exp(-Se*e_Se)
        
        # gowth of Se
        u_Se        = u_max_Se * (L /(k_L_Se + L))
        dSedt   = Se * (u_Se -k_death_Se)

        return jnp.array([dSedt])
    
    
class Simulation_Data_minimal_Suc_feed(eqx.Module):
    """
    The complete Ruoff (2003) glycolysis model with **trainable** k5.

    State vector (7 components)
    ---------------------------
    Se    — autotroph
    Um    — heterotroph
    Suc   — Sucrose
    
    Parameter vector (13 components)
    ---------------------------------
    u_max_Se, u_max_Um, k_L_se, k_Suc_se, 

    ``k5`` controls the ATP-degradation rate  (-k5 · A3 in dA3/dt).
    It is kept as a free/learnable parameter here to allow the ODE to
    compensate for the missing term.  In ``ODEcore`` this term is removed
    entirely, with a neural network learning it instead.

    Parameters
    ----------
    weights : jax.Array, shape (13,)
        All 13 model parameters, including k5.

    Examples
    --------
    >>> ode = ODE(weights=RUOFF_PARAMS)
    >>> ode(t=0.0, y=Y0_RUOFF, args=None)   # returns shape (7,)
    """

    weights: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        Se, Um, Suc = variables
        u_max_Se, u_max_Um, k_L_Se, k_Suc_Se, k_Suc_Um, y_Um_Suc, L_0, e_Se, e_UM, k_death_Se, k_death_Um, Suc_feed = self.weights
        
        # derived
        L           = L_0*jnp.exp(-Se*e_Se - Um*e_UM)
        
        # gowth of Se
        u_Se        = u_max_Se * (L /(k_L_Se + L))
          
        # growth of Um
        u_Um        = u_max_Um * (Suc /(k_Suc_Um + Suc))
        
        # Sucrose export by Se
        Suc_exp_Se  = Se * u_Se * k_Suc_Se
    
        # Sucrose import by Um
        Suc_imp_Um  =Um * u_Um/y_Um_Suc
        
        # change of variables
        dSedt   = Se * (u_Se -k_death_Se)
        dUmdt   = Um * (u_Um -k_death_Um)
        dSucdt  = Suc_exp_Se - Suc_imp_Um + Suc_feed
        

        return jnp.array([dSedt, dUmdt, dSucdt])


class Community_Final_Model (eqx.Module):
    
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
    parameters_Sc: Array
    parameters_abiotic: Array

    def __call__(self, t: float, variables: Array, args) -> Array:
        
        # parameters and variables:
        # ----------------------------------------------------
        Se,Um,Sc,Suc,N,P1,Met1,Met2,pH,B = variables
        u_max_Se, k_L_Se, e_Se,  k_N_Se, Y_Se_N, k_death_Se, alpha_Suc_Se, k_Met1_Se, Y_Se_Met1, beta_Met2_Se, k_P1_Met2_Se, K_I_pH_Suc_exp = self.parameters_Se
        u_max_Um,k_Suc_Um, Y_Um_Suc, k_N_Um,Y_Um_N, k_death_Um, e_Um, beta_Met1_Um, K_I_Met1_Um, K_I_Met2_Um = self.parameters_Um
        u_max_Sc, k_Suc_Sc, Y_Sc_Suc, k_N_Sc, Y_N_Sc, k_death_Sc, e_Sc, alpha_H_Sc = self.parameters_Sc
        L_0, e_medium, K_B,= self.parameters_abiotic
    
        # derived
        L = L_0 * jnp.exp(-Se*e_Se - Um*e_Um - Sc*e_Sc - e_medium)
        
        # growth of Se
        u_Se    = u_max_Se * (L /(k_L_Se + L)) * (N/(k_N_Se+N)) * (1+(Met1/(k_Met1_Se+Met1)))
        Met1_imp_Se = Se * u_Se * (Met1/(k_Met1_Se+Met1))/Y_Se_Met1
        Met2_exp_Se = Se * beta_Met2_Se * P1
        Suc_exp_Se  = Se * alpha_Suc_Se * L * ((K_I_pH_Suc_exp / K_I_pH_Suc_exp) + pH)
    
        #Saccharomyces Cerevisiae
        u_Sc           = u_max_Sc * (Suc /(k_Suc_Sc + Suc)) * (N/(k_N_Sc+N))
        Suc_imp_Sc     = u_Sc * Sc * 1/Y_Sc_Suc
        exp_pH_Sc       = Sc * u_Sc * alpha_H_Sc
        
        # growth of Um
        u_Um    = u_max_Um * (Suc /(k_Suc_Um + Suc)) * (N/(k_N_Um+N)) * (K_I_Met2_Um/(K_I_Met2_Um + Met2))
        Suc_imp_Um = u_Um * Um * 1/Y_Um_Suc
        Met1_exp_Um = Um * u_Um * beta_Met1_Um * (K_I_Met1_Um/(Met1+K_I_Met1_Um))
        
        # change of variables
        #---------------------------------------------------------
        # microbes
        dSedt       = Se * (u_Se - k_death_Se)
        dUmdt       = Um * (u_Um - k_death_Um)
        dScdt       = Sc * (u_Sc - k_death_Sc)
        
        # paramters, metabolites, etc. (ressources)
        dSucdt      = Suc_exp_Se -Suc_imp_Um
        dNdt        = -u_Se*Se * 1/Y_Se_N - u_Um*Um * 1/Y_Um_N
        dP1dt       = 0.0
        dMet1dt     = Met1_exp_Um - Met1_imp_Se
        dMet2dt     = Met2_exp_Se
        dBdt        = - exp_pH_Sc*((B+K_B)/K_B)
        dpHdt       = exp_pH_Sc*(K_B/(K_B+B))
        

        return jnp.array([dSedt,dUmdt,dScdt,dSucdt,dNdt,dP1dt,dMet1dt,dMet2dt,dpHdt,dBdt])


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



class UDE_Community(eqx.Module):
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
        
        scaling = jnp.ones_like(y)
        scaling = scaling.at[0:3].set(y[0:3])
        return ode_out + self.lambda_ * scaling * nde_out
    
    
    
    
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
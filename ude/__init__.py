"""
ude — Universal Differential Equations for Systems Biology
==========================================================

A modular library for building, training, and evaluating hybrid
mechanistic/neural-network ODE models.

Public API
----------
Models:
    NDE        — pure neural differential equation
    ODE        — full Ruoff glycolysis model (trainable k5)
    ODEcore    — mechanistic skeleton (missing the k5 term)
    UDE        — hybrid = ODEcore + NDE

Solver:
    OdeProblem — wraps any RHS into a diffrax-based ODE solver

Training:
    train_model   — two-phase curriculum training loop

Data:
    simulate_reference — generate ground-truth glycolysis trajectories
    prepare_data       — split, noise, format for training

Visualisation:
    plot_comparison — ODE vs NDE vs UDE vs ground truth

Example
-------
>>> from ude import ODEcore, NDE, UDE, OdeProblem, train_model
>>> import jax
>>> key = jax.random.PRNGKey(0)
>>> ode = ODEcore(weights=RUOFF_PARAMS_CORE)
>>> nde = NDE(in_size=7, out_size=7, width_size=32, depth=2, key=key)
>>> model = OdeProblem(UDE(ode, nde), full_state=True)
>>> trained = train_model(model, data, y0, ts_train, loader_key=key)
"""
import pandas as pd
from ude.models import NDE, UDE, ODE_LV_core, UDE_Community
from ude.solver import OdeProblem
from ude.training import *
from ude.data import *
### change this when ready
# __all__ = [
    
#     # Models
#     "NDE", "UDE","ODE_LV_core"
#     # Solver
#     "OdeProblem",
#     # Training
#     "train_model", "dataloader",
#     # Data
#     prepare_data,PARAMETERSET_COMMUNITY_Se,PARAMETERSET_COMMUNITY_Um, PARAMETERSET_COMMUNITY_Sc, 
#     PARAMETERSET_COMMUNITY_abiotic, PARAMETERSET_COMMUNITY_abiotic_feed, START_CONDITION_COMMUNITY, STARTCONDITION_Axenic_Se,
#     STARTCONDITION_Axenic_Um, STARTCONDITION_Axenic_Sc, STARTCONDITION_co_Se_Um, STARTCONDITION_co_Se_Sc, STARTCONDITION_co_Um_Sc
#     # Visualisation
#     "plot_comparison",  "plot_learned_term",
# ]

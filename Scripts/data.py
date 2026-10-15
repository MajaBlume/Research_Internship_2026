"""
ude.data
========

Reference parameters, initial conditions, data generation, and
train/test splitting for the Ruoff glycolysis model.

Constants
---------
RUOFF_PARAMS      — full 13-parameter vector (includes k5)
RUOFF_PARAMS_CORE — 12-parameter vector (k5 removed, for ODEcore / UDE)
RUOFF_PARAMS_K5_ZERO — full vector with k5 set to 0 (ODE baseline)
Y0_RUOFF          — reference initial conditions
OBS_IDX           — indices of observable species [N2=4, A3=5]

Functions
---------
simulate_reference  — run the ground-truth ODE and return trajectory
prepare_data        — split, add noise, format arrays for training

References
----------
Ruoff, P. et al. (2003). The Goodwin model. J. Theor. Biol., 209(1), 29–42.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax

# Parametersets
# -------------------------------------------
# PARAMETERSET_COMMUNITY_Se
# PARAMETERSET_COMMUNITY_Um
# PARAMETERSET_COMMUNITY_Sc
# PARAMETERSET_COMMUNITY_abiotic
# PARAMETERSET_COMMUNITY_abiotic_feed
# START_CONDITION_COMMUNITY
# STARTCONDITION_Axenic_Se
# STARTCONDITION_Axenic_Um
# STARTCONDITION_Axenic_Sc
# STARTCONDITION_co_Se_Um
# STARTCONDITION_co_Se_Sc
# STARTCONDITION_co_Um_Sc



PARAMETERSET_COMMUNITY_Se = [
    # S. elongatus
    0.08,  # u_max_Se
    0.01,   # k_L_Se
    3.0,    # e_Se
    0.6,    # k_N_Se
    8.0,    # Y_Se_N
    0.005,  # k_death_Se
    0.4,    # beta_Suc_Se
    0.2,    # k_Met1_Se
    0.2,    # beta_Met2_Se
    0.2,    # k_P1_Met2_Se
    0.2,    # K_I_pH_Suc_exp   
]

PARAMETERSET_COMMUNITY_Um = [
    # U.maydis
    0.2,    # u_max_Um
    0.01,   # k_Suc_Um
    0.15,   # Y_Um_Suc
    0.1,    # k_N_Um
    10.0,   # Y_N_Um
    0.005,  # k_death_Um
    0.3,    # e_Um
    0.2,    # beta_Met1_Um
    0.2,    # K_I_Met1_Um
    0.2,    # K_I_Met2_Um
    0.2     # k_Met2_Um  
]

PARAMETERSET_COMMUNITY_Sc = [
    0.4,     #u_max_Sc
    0.1,     # k_Suc_Sc
    0.12,    # Y_Sc_Suc
    0.12,    # k_N_Sc
    9,       # Y_N_Sc
    0.007,   # k_death_Sc
    1,       # e_Sc 
    0.1,     # alpha_H_Sc
]

PARAMETERSET_COMMUNITY_abiotic = [
    2,     # L0
    0,     # k_Suc_Sc  
]

PARAMETERSET_COMMUNITY_abiotic_feed = [
    2,          # L0
    0.0208,     # 3.5/168
]

# Starting Conditions
# -------------------------------------------

START_CONDITION_COMMUNITY=[  
    0.1,    # S. elongatus
    0.05,    # U.maydis
    0.05,    # S.cerevisiae
    0.01,   # Sucrose
    3,      # Nitrogen
    0.0,    # Parameter1
    0.0,    # Metabolite1
    0.0,    # Metabolite2
    0.0,    # pH equivalent (simplified)    
]

STARTCONDITION_Axenic_Se=[  
    0.1,    # S. elongatus
    0.0,    # U.maydis
    0.0,    # S.cerevisiae
    0.01,   # Sucrose
    3,      # Nitrogen
    0.0,    # Parameter1
    0.0,    # Metabolite1
    0.0,    # Metabolite2
    0.0,    # pH equivalent (simplified)    
]

STARTCONDITION_Axenic_Um = [
    0.0,    # S. elongatus
    0.05,    # U.maydis
    0.0,    # S.cerevisiae
    0.001,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
    0.0,     # pH equivalent (simplified) 
]

STARTCONDITION_Axenic_Sc = [
    0.0,    # S. elongatus
    0.0,    # U.maydis
    0.05,    # S.cerevisiae
    0.001,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
    0.0,     # pH equivalent (simplified) 
]

STARTCONDITION_co_Se_Um = [
    0.1,    # S. elongatus
    0.05,    # U.maydis
    0.00,    # S.cerevisiae
    0.001,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
    0.0,     # pH equivalent (simplified) 
]

STARTCONDITION_co_Um_Sc = [
    0.0,    # S. elongatus
    0.05,    # U.maydis
    0.05,    # S.cerevisiae
    0.001,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
    0.0,     # pH equivalent (simplified) 
]

STARTCONDITION_co_Se_Sc = [
    0.1,    # S. elongatus
    0.0,    # U.maydis
    0.05,    # S.cerevisiae
    0.001,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
    0.0,     # pH equivalent (simplified) 
]


COCULTURE_PARAMS:jax.Array = jnp.array([
1.89*10**(-5)*3600,     # u_max_Se, 1/s
6.71*10**-5*3600,       # u_max_Um, 1/s
0.00035,                # k_L_Se, (fitted)
1,                # k_Suc_Se, Kg/m^-3
3.6,                    # k_Suc_Um, Kg/m^-3
0.34,                   # y_Um_Suc, g Biomass/g Suc
200,                    # L_0, [Einheit??]
0.2,                    # e_Se, 
0.1,                    # e_UM,
0.01,                    # death_rate Se this work
0.01,                    # death_rate Um this work
])

COCULTURE_Feed_PARAMS:jax.Array = jnp.array([
1.89*10**(-5)*3600,     # u_max_Se, 1/s
6.71*10**-5*3600,       # u_max_Um, 1/s
0.00035,                # k_L_Se, (fitted)
1,                      # k_Suc_Se, Kg/m^-3
3.6,                    # k_Suc_Um, Kg/m^-3
0.34,                   # y_Um_Suc, g Biomass/g Suc
200,                    # L_0, [Einheit??]
0.2,                    # e_Se, 
0.1,                    # e_UM,
0.01,                   # death_rate Se this work
0.01,                   # death_rate Um this work
0.3,                   # Suc_feed
])


    # # Add measurement noise if a key is provided
    # if noise_key is not None and noise_std > 0.0:
    #     k1, k2 = jax.random.split(noise_key)
    #     obs_train = obs_train + noise_std * jax.random.normal(k1, obs_train.shape)
    #     obs_test  = obs_test  + noise_std * jax.random.normal(k2, obs_test.shape)

    # y0_full = raw[:, 0, :]          # (1, 7)
    # y0_obs  = y0_full[:, OBS_IDX]   # (1, 2)

    # return ts_full, ts_train, ts_test, obs_train, obs_test, y0_full, y0_obs

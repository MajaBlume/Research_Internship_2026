"""
Here all functions and parameters to prepare data is collected
1. Parameters and starting conditions

"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import pandas as pd

PARAMETERSET_COMMUNITY_Se = [
    # S. elongatus
    0.15,       # u_max_Se
    1,          # k_L_Se
    1,          # e_Se
    0.3,        # k_N_Se
    10.0,       # Y_Se_N
    0.005,      # k_death_Se
    0.1,        # alpha_Suc_Se
    0.01,        # k_Met1_Se
    7,          # Y_Se_Met1
    0.001,       # beta_Met2_Se
] 

PARAMETERSET_COMMUNITY_Um = [
    # U.maydis
    0.1,    # u_max_Um
    0.15,   # k_Suc_Um
    0.2,   # Y_Um_Suc
    0.1,    # k_N_Um
    5.0,   # Y_N_Um
    0.002,  # k_death_Um
    0.3,    # e_Um
    0.9,    # beta_Met1_Um
    0.5,    # K_I_Met1_Um
    0.08,    # K_I_Met2_Um
]

PARAMETERSET_COMMUNITY_abiotic = [
    1,     # L0
    0.1,   # e_medium
]

STARTCONDITION_Medium = [
    0.0,    # S. elongatus
    0.00,   # U.maydis
    3,   # Sucrose
    3,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]


STARTCONDITION_COMMUNITY=[  
    0.1,    # S. elongatus
    0.01,    # U.maydis
    0.01,   # Sucrose
    5,      # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Axenic_Um = [
    0.0,     # S. elongatus
    0.01,    # U.maydis
    3,       # Sucrose
    5,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Coculture_Se_Um = [
    0.1,     # S. elongatus
    0.01,    # U.maydis
    0.0,     # Sucrose
    5,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Coculture_Se_Um = [
    0.1,     # S. elongatus
    0.01,    # U.maydis
    0.0,     # Sucrose
    5,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Coculture_Se_Um_P1 = [
    0.1,     # S. elongatus
    0.01,    # U.maydis
    0.0,     # Sucrose
    5,       # Nitrogen
    1.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2 
]

STARTCONDITION_Axenic_Se = [
    0.1,     # S. elongatus
    0.0,    # U.maydis
    0.0,     # Sucrose
    5,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Axenic_Se_N_low = [
    0.1,     # S. elongatus
    0.00,    # U.maydis
    0.0,     # Sucrose
    1,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2 
]

STARTCONDITION_Axenic_Se_Met1_Met2 = [
    0.1,     # S. elongatus
    0.00,    # U.maydis
    0.0,     # Sucrose
    5.0,     # Nitrogen
    1.0,     # Parameter1
    0.2,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Axenic_Se_P1 = [
    0.1,     # S. elongatus
    0.00,    # U.maydis
    0.0,     # Sucrose
    5.0,     # Nitrogen
    1.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Axenic_Um_N_low = [
    0.0,     # S. elongatus
    0.01,    # U.maydis
    3,       # Sucrose
    1,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

STARTCONDITION_Axenic_Um_Met2 = [
    0.0,     # S. elongatus
    0.01,    # U.maydis
    3,       # Sucrose
    5,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.3,     # Metabolite2
]

STARTCONDITION_Coculture_low_N = [
    0.1,     # S. elongatus
    0.01,    # U.maydis
    0.0,     # Sucrose
    1,       # Nitrogen
    0.0,     # Parameter1
    0.0,     # Metabolite1
    0.0,     # Metabolite2
]

start_conditions_dict = {
    "Medium": STARTCONDITION_Medium,
    "Axenic_Se": STARTCONDITION_Axenic_Se,
    "Axenic_Se_N_low": STARTCONDITION_Axenic_Se_N_low,
    "Axenic_Um": STARTCONDITION_Axenic_Um,
    "Axenic_Um_N_low": STARTCONDITION_Axenic_Um_N_low,
    "Axenic_Se_Met1_Met2": STARTCONDITION_Axenic_Se_Met1_Met2,
    "Axenic_Se_P1": STARTCONDITION_Axenic_Se_P1,
    "Axenic_Um_Met2": STARTCONDITION_Axenic_Um_Met2,
    "Coculture_Se_Um": STARTCONDITION_Coculture_Se_Um,
    "Coculture_Se_Um_P1": STARTCONDITION_Coculture_Se_Um_P1,
    "Coculture_low_N": STARTCONDITION_Coculture_low_N,
}
all=["Se","Um","Suc","N","P1","Met1","Met2"]
start_conditions_df = pd.DataFrame.from_dict(start_conditions_dict, orient="index", columns=["Se","Um","Suc","N","P1","Met1","Met2"])

def get_timepoint_indices (time_in_hours: list):
    indices=[]
    for x, timepoint in enumerate(time_in_hours):
        indices.append(int(200/150*timepoint))
    return jnp.array(indices)

def add_noise (data:Array, std_dev, key):
    return data*(1+std_dev*jr.normal(key,data.shape)) 


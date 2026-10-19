""" Collection of all visualization methods
"""

def display_ODE_coculture(t_steps, ode_solution, display_variabs, ax, line="-", std=0):
    
    variables_dict = {
        "Se":  ["S. elongatus", 0, "blue"],
        "Um":  ["U. maydis", 1, "green"],
        "Suc": ["Sucrose",2, "gray"], 
        "N" :  ["Nitrogen",3, "red"],
        "P1":  ["Parameter 1",4, "brown"],
        "Met1": ["Metabolite 1",5, "yellow"], 
        "Met2": ["Metabolite 2",6, "olive"],
    }
    
    for var in display_variabs:
        ax.plot(
            t_steps,
            ode_solution[:,variables_dict[var][1]],
            color= variables_dict[var][2],
            label= variables_dict[var][0],
            linestyle=line
        )
        ax.fill_between (
            t_steps,
            ode_solution[:,variables_dict[var][1]]*(1+std),
            ode_solution[:,variables_dict[var][1]]*(1-std),
            alpha=0.3,
            linewidth=0,
            color= variables_dict[var][2],
            label= "_nolegend_",        # this makes no label appear in the legend for the fill between
            linestyle=line
        )
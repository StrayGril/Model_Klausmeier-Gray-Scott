import numpy as np
import matplotlib.pyplot as plt
from pipeline.model_core import (
    make_grid,
    dirichlet_boundary_mask,
    homogeneous_state,
    precompute_diffusion,
    step_reaction_diffusion
)

# --------------------------------------------------
# Warunki początkowe
# --------------------------------------------------
def initial_conditions(Nx, Ny, a, m, brzeg, noise=1e-3):

    u_star, v_star = homogeneous_state(a, m)

    u = u_star * np.ones(Nx * Ny)
    v = v_star * np.ones(Nx * Ny)

    rng = np.random.default_rng()

    u += noise * rng.standard_normal(Nx * Ny)
    v += noise * rng.standard_normal(Nx * Ny)

    u[brzeg] = 0
    v[brzeg] = 0

    return u, v


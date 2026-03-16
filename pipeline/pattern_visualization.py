import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from numpy.random import default_rng

# Dodanie katalogu głównego repo do ścieżki
sys.path.append(os.path.abspath(".."))


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


# --------------------------------------------------
# Symulacja wzorów
# --------------------------------------------------
def simulate_patterns(a, m, d1, d2, Lx, Ly, Nx, Ny, T, wykres):
    # tworzy wszystkie potrzebne macierz
    # korzysta z ww funkcji
    # liczy aż powstaną wzory (do kiedy? obecnie T razy)
    # zwraca dane do wykresu (wykres= "u", "v" lub "uv" lub brak, ze wtedy robi wyrkesy dla u lub v lub obu

    x, y, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    ht = 0.025
    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)

    u_0, v_0 = initial_conditions(Nx, Ny, a, m, brzeg, noise=1e-2) #zmienilam 3 na 2

    u_curr, v_curr = u_0.copy(), v_0.copy()

    u_t1 = None
    v_t1 = None
    u_tmid = None
    v_tmid = None

    for i in range(T):
        u_curr, v_curr = step_reaction_diffusion(u_curr, v_curr, a, m, ht, lu_Au, lu_Av, brzeg)


        if i == 1:
            u_t1 = u_curr.copy()
            v_t1 = v_curr.copy()

        if i == int(T / 2):
            u_tmid = u_curr.copy()
            v_tmid = v_curr.copy()

    if "v" in wykres:

        fig, axs = plt.subplots(2,2, figsize=(10,8))

        levelsv = np.linspace(min(v_0.min(), v_t1.min(), v_tmid.min(), v_curr.min()),
                              max(v_0.max(), v_t1.max(), v_tmid.max(), v_curr.max()), 50)

        axs[0,0].contourf(X, Y, v_0.reshape(Ny, Nx), levels=levelsv, cmap='viridis')
        axs[0,0].set_title("v(t=0)")

        axs[0,1].contourf(X, Y, v_t1.reshape(Ny, Nx), levels=levelsv, cmap='viridis')
        axs[0,1].set_title("v(t=1)")

        axs[1,0].contourf(X, Y, v_tmid.reshape(Ny, Nx), levels=levelsv, cmap='viridis')
        axs[1,0].set_title("v(t=T/2)")

        im = axs[1,1].contourf(X, Y, v_curr.reshape(Ny, Nx), levels=levelsv, cmap='viridis')
        axs[1,1].set_title("v(t=T)")

        fig.colorbar(im, ax=axs)
        #plt.tight_layout()
        plt.show()


    if "u" in wykres:

        fig, axs = plt.subplots(2,2, figsize=(10,8))

        levelsu = np.linspace(min(u_0.min(), u_t1.min(), u_tmid.min(), u_curr.min()),
                              max(u_0.max(), u_t1.max(), u_tmid.max(), u_curr.max()), 50)

        axs[0,0].contourf(X, Y, u_0.reshape(Ny, Nx), levels=levelsu, cmap='viridis')
        axs[0,0].set_title("u(t=0)")

        axs[0,1].contourf(X, Y, u_t1.reshape(Ny, Nx), levels=levelsu, cmap='viridis')
        axs[0,1].set_title("u(t=1)")

        axs[1,0].contourf(X, Y, u_tmid.reshape(Ny, Nx), levels=levelsu, cmap='viridis')
        axs[1,0].set_title("u(t=T/2)")

        im = axs[1,1].contourf(X, Y, u_curr.reshape(Ny, Nx), levels=levelsu, cmap='viridis')
        axs[1,1].set_title("u(t=T)")

        fig.colorbar(im, ax=axs)
        #plt.tight_layout()
        plt.show()

    return u_curr, v_curr




# Wykresy z symulacji - robia sie w symualcji obecnie zeby podgladac jej stan bo ona Robi Rzeczy^TM
def plot_patterns(wynik_simulate_patterns):
    # robi wykres z wektorów u_fin, v_fin
    ...


# =====================================
# Próba 1 uzyskania wzorów (wychodzi dziwnie ale sa to jakies wzory?)

a, m, d1, d2 = 1, 0.45, 1, 0.02

L, N = 10, 30

simulate_patterns(a, m, d1, d2, L, L, N, N, T=500, wykres="v")




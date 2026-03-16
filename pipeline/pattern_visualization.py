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

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        dane = [v_0, v_t1, v_tmid, v_curr]
        tytuly = ["v(t=0)", "v(t=1)", "v(t=T/2)", "v(t=T)"]

        levelsv = np.linspace(
            min(d.min() for d in dane),
            max(d.max() for d in dane),
            50
        )

        for ax, d, t in zip(axs.flat, dane, tytuly):
            im = ax.contourf(X, Y, d.reshape(Ny, Nx), levels=levelsv, cmap="viridis")
            ax.set_title(t)

        fig.colorbar(im, ax=axs)
        #plt.tight_layout()
        plt.show()

    if "u" in wykres:

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        dane = [u_0, u_t1, u_tmid, u_curr]
        tytuly = ["u(t=0)", "u(t=1)", "u(t=T/2)", "u(t=T)"]

        levelsu = np.linspace(
            min(d.min() for d in dane),
            max(d.max() for d in dane),
            50
        )

        for ax, d, t in zip(axs.flat, dane, tytuly):
            im = ax.contourf(X, Y, d.reshape(Ny, Nx), levels=levelsu, cmap="viridis")
            ax.set_title(t)

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




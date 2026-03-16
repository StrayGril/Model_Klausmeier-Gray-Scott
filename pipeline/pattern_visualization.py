import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
def simulate_patterns(a, m, d1, d2, Lx, Ly, Nx, Ny, T):
    """
        Przeprowadza symulację o podanych parametrach na T powtórzeń.
        Parametry:
            ...
        Zwraca słownik danych do wykresu, najważniejsze to stany końcowe symulacji: macierze uT i vT.
    """
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


    return {"X": X,
        "Y": Y,
        "u0": u_0,
        "v0": v_0,
        "u1": u_t1,
        "v1": v_t1,
        "umid": u_tmid,
        "vmid": v_tmid,
        "uT": u_curr,
        "vT": v_curr}



# --------------------------------------------------
# Wykresy z symulacji
# --------------------------------------------------
def plot_patterns(sim_data, wykres="uv"):
    """
        Rysuje wykresy symulacji w chwilach: 0, 1, T/2, T.

        Parametry:
        sim_data - wynik funkcji simulate_patterns
        wykres : string "u", "v", "uv", które wykresy pokazać
    """
    X, Y = sim_data["X"], sim_data["Y"]
    Ny, Nx = X.shape

    states = {
        "0": (sim_data["u0"], sim_data["v0"]),
        "1": (sim_data["u1"], sim_data["v1"]),
        "T/2": (sim_data["umid"], sim_data["vmid"]),
        "T": (sim_data["uT"], sim_data["vT"])}

    if "u" in wykres: # wykres wody

        fig, axs = plt.subplots(2,2, figsize=(10,8))
        dane = [states[t][0] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (u, _)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, u.reshape(Ny, Nx), levels=levels, cmap="Spectral")
            ax.set_title(f"u(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()


    if "v" in wykres: # wykres biomasy

        fig, axs = plt.subplots(2,2, figsize=(10,8))
        dane = [states[t][1] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (_, v)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, v.reshape(Ny, Nx), levels=levels, cmap="RdYlGn")
            ax.set_title(f"v(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()


# ===============================================
# Wychodza wzory!! dla np L, N = 20, 60, T=8000

a, m, d1, d2 = 1, 0.45, 1, 0.02

L, N = 20, 60

wynik = simulate_patterns(a, m, d1, d2, L, L, N, N, T=8000)
plot_patterns(wynik, wykres="uv")


# DALEJ:
# sprobowac dac w, n zamiast u, v i - czy zadziala dla wielowymiarowego

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

# ============
# Plan funkcji

def simulate_patterns(a, m, d1, d2, Lx, Ly, Nx, Ny):
    # tworzy wszystkie potrzebne macierz
    # korzysta z ww funkcji
    # liczy aż powstaną wzory (kiedy?)
    # zwraca dane do wykresu
    ...

def plot_patterns(wynik_simulate_patterns):
    # robi wykres z wektorów u_fin, v_fin
    ...


# =====================================
# Próba 1 uzyskania wzorów (wychodzi dziwnie ok)

a, m, d1, d2 = 1, 0.45, 1, 0.02

L, N = 10, 30
x, y, X, Y, h = make_grid(L, L, N, N)

brzeg = dirichlet_boundary_mask(X, Y, L, L)

ht = 0.025
lu_Au, lu_Av = precompute_diffusion(N, N, h, ht, d1, d2)


# tworzymy u, v początkowe - ze stanu rownowagi + szum
us, vs = homogeneous_state(a, m)
u_0 = np.full_like(X, us).flatten()
v_0 = np.full_like(X, vs).flatten()

rng = np.random.default_rng()
u_0 += 1e-2 * rng.standard_normal(N * N).flatten()
v_0 += 1e-2 * rng.standard_normal(N * N).flatten()

u_curr, v_curr = u_0.copy(), v_0.copy()

# symulacja T razy i wykresy
T = 500
for i in range(T):
    u_curr, v_curr = step_reaction_diffusion(u_curr, v_curr, a, m, ht, lu_Au, lu_Av, brzeg)
    if i in [1, T/2]:
        levelsv = np.linspace(v_curr.min(), v_curr.max(), 50)

        if i==1:
            pv0 = plt.contourf(X, Y, v_0.reshape(N, N), levels=levelsv, cmap='viridis')
            cbar = plt.colorbar(pv0)
            plt.title("Warunek początkowy v_0")
            plt.show()

        pvc = plt.contourf(X, Y, v_curr.reshape(N, N), levels=levelsv, cmap='viridis')
        cbar = plt.colorbar(pvc)
        plt.title(f"v_curr: {i}")
        plt.show()


v_curr = v_curr.reshape(N, N)

levelsv = np.linspace(v_curr.min(), v_curr.max(), 50)

pvc = plt.contourf(X, Y, v_curr, levels=levelsv, cmap='viridis')
cbar = plt.colorbar(pvc)
plt.title("v_fin")
plt.show()

plt.plot(v_curr)
plt.show()



# Wykresy wody niewazne na razie
if False:
    u_curr = u_curr.reshape(N, N)

    levelsu = np.linspace(u_curr.min(), u_curr.max(), 50)

    pu0 = plt.contourf(X, Y, u_0.reshape(N, N), levels=levelsu, cmap='viridis')
    cbar = plt.colorbar(pu0)
    plt.title("Warunek początkowy u_0")
    plt.show()

    puc = plt.contourf(X, Y, u_curr, levels=levelsu, cmap='viridis')
    cbar = plt.colorbar(puc)
    plt.title("u_curr")
    plt.show()


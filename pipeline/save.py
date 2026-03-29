import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.patterns import simulate_patterns, plot_matrix, cmap_v


# Generating and saving matrices
def save_as_npz(
        file_name,
        a_vector,
        m_vector,
        d1_vector,
        d2_vector,
        lx=60,
        ly=60,
        nx=100,
        ny=100,
        T=10000,
        ht=0.025,
        folder="wykresy_bez_etykiet",
        verbose=False,
):
    """
    Runs multiple simulations for given parameter sets and saves results to a .npz file.
    If a simulation becomes unstable or blows up, the last stable state is returned.
    Saves a compressed .npz file containing matrices (U, V) and corresponding parameter sets (a, m, d1, d2) and an empty set for patterns.

    Parameters
    file_name : str
        Name of the output file (without .npz extension).
    a, m, d1, d2 : array-like
        Vectors of model parameters (must be equal length).
    lx, ly : float, optional
        Domain sizes. Default: 60.
    nx, ny : int, optional
        Number of grid points. Default: 100.
    T : int, optional
        Maximum number of time steps. Default: 10000.
    ht : float, optional
        Time step. Default: 0.025
    folder : str, optional
        Target directory for the saved .npz file. Default: "wykresy_bez_etykiet".
    verbose : bool, optional
        If True, prints diagnostic information about simulation progress and stability.

    Returns: None
    """

    length = len(a_vector)

    if ((len(m_vector) != length)
            or (len(d1_vector) != length)
            or (len(d2_vector) != length)
    ):
        raise ValueError("Vectors' lengths unequal")

    U = []
    V = []
    a_ok = []
    m_ok = []
    d1_ok = []
    d2_ok = []

    os.makedirs(folder, exist_ok=True)

    old_settings = np.seterr(over='ignore', invalid='ignore', divide='ignore') #ignore

    try:
        for i in range(length):  # symulacja dla kolejnych parametrow
            try:
                u, v = simulate_patterns(
                    a_vector[i],
                    m_vector[i],
                    d1_vector[i],
                    d2_vector[i],
                    lx=lx,
                    ly=ly,
                    nx=nx,
                    ny=ny,
                    T=T,
                    ht=ht,
                    return_matrices=True
                )

                if (not np.all(np.isfinite(u))) or (not np.all(np.isfinite(v))):
                    if verbose:
                        print(f"Pomijam i={i}: wynik ma NaN/inf")
                    continue

                # chcemy macierze czy wektory? obie czy v?
                U.append(u)
                V.append(v)
                a_ok.append(a_vector[i])
                m_ok.append(m_vector[i])
                d1_ok.append(d1_vector[i])
                d2_ok.append(d2_vector[i])

            except Exception as e:
                if verbose:
                    print(
                        f"Błąd dla i={i}, a={a_vector[i]}, m={m_vector[i]}, "
                        f"d1={d1_vector[i]}, d2={d2_vector[i]}: {e}"
                    )
                continue
    finally:
        np.seterr(**old_settings)

    sciezka = os.path.join(folder, f"{file_name}.npz")

    np.savez_compressed(
        sciezka,
        U=np.array(U),
        V=np.array(V),
        a=np.array(a_ok),
        m=np.array(m_ok),
        d1=np.array(d1_ok),
        d2=np.array(d2_ok),
        patterns=np.full(len(a_ok), -1, dtype=int)
    )

    if verbose:
        print("Saving complete.")


# Defining the patterns from npz file by hand
def define_patterns(file_name, folder="wykresy_etykiety", folder_old="wykresy_bez_etykiet", cmap=None):
    """
    Interactively labels Turing patterns and saves back to an .npz file.

    Logic
        1. Displays each matrix
        2. User manually assigns a category (0. nothing, 1. spots, 2. stripes, 3. labyrinths, 4. gaps, 5. something else) or quits (q)

    It allows for resuming previous work by skipping already labeled matrices.

    Parameters
    file_name : str
        The name of the .npz file to load (without extension).

    folder : str, optional
        The destination directory. Default: "wykresy_etykiety".

    folder_old : str, optional
        The source directory. Default is "wykresy_bez_etykiet".

    cmap : matplotlib.colors.Colormap, optional
        Colormap used for displaying the matrices. Defaults to the globally defined 'cmap_v'.

    Returns: None

    Progress is saved only after the loop finishes or is interrupted by 'q'.
    """
    our_path = os.path.join(folder_old, f"{file_name}.npz")

    with np.load(our_path, allow_pickle=True) as loader:
        dane = dict(loader)

    if cmap is None:
        cmap = cmap_v

    length = len(dane["V"])
    patterns = dane["patterns"].copy()

    for i in range(length):
        # pomijamy te, które już mają etykiete
        if patterns[i] != -1:
            continue

        title = f"{file_name}, i={i}"
        plot_matrix(dane["V"][i], plot_title=title, show=False, cmap=cmap)

        ans = input("0. nothing, 1. spots, 2. stripes, 3. labyrinths, 4. gaps, 5. something else (d=delete)(q=quit): ")

        if ans.lower() == 'q':
            plt.close()
            break
        if ans.lower() == 'd':
            patterns[i] = 99
            plt.close()
            continue

        try:
            patterns[i] = int(ans)
        except ValueError:
            print("Skipped (unknown input).")

        plt.close()

    patterns = np.array(patterns)
    keep_mask = patterns < 99

    for key in ["U", "V", "a", "m", "d1", "d2"]:
        dane[key] = dane[key][keep_mask]

    dane["patterns"] = patterns[keep_mask]

    print(f"Deleted {np.sum(~keep_mask)}/{length} matrices.")


    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, f"{file_name}.npz")
    np.savez_compressed(output_path, **dane)

    length = len(dane["patterns"])
    stayed = sum(1 for x in dane["patterns"] if x != -1)
    print(f"End of file. Patterns are defined on {stayed}/{length} matrices.")
    print(f"File saved to: {output_path}")


# konwersja do csv z (samych parametrów)
def convert_to_csv(npz_file_name):
    dane = np.load(f"{npz_file_name}.npz", allow_pickle=True)

    # Tworzymy słownik tylko z tych danych, które chcemy w tabeli
    tabela = {
        'a': dane['a'],
        'm': dane['m'],
        'd1': dane['d1'],
        'd2': dane['d2'],
        'pattern': dane['patterns']
    }

    df = pd.DataFrame(tabela)

    output_name = f"{npz_file_name}.csv"
    df.to_csv(output_name, index=False)

    print(f"Tabela została zapisana do pliku: {output_name}.")
    return df

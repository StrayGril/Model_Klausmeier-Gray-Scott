import numpy as np
import pandas as pd
import joblib

# ============================================================
# WCZYTAJ MODEL
# ============================================================

model = joblib.load('model1_klasyfikator.pkl')
class_names = np.load('model1_class_names.npy', allow_pickle=True)

# ============================================================
# GENEROWANIE PARAMETRÓW Z TWOICH ZAKRESÓW
# ============================================================

def generuj_parametry(n_samples=100000, seed=None):
    rng = np.random.default_rng(seed)

    dane = np.empty((n_samples, 4))  # kolumny: a, m, d1, d2

    for i in range(n_samples):
        # przypadek 1: d2 = 0.02 z prawdopodobieństwem 60%
        if rng.random() < 0.6:
            d2 = 0.02

            # d1 z {1,...,39,100}
            d1 = rng.choice(np.append(np.arange(1, 40), 100))

            # a z (0, 5] co 0.02
            a = rng.choice(np.arange(0.02, 5 + 0.001, 0.02))

            # ograniczenie na m
            if d1 < 100:
                m_max = min(1.5, a / 2)
            else:
                m_max = min(1.0, a / 2)

            # m z siatki co 0.01, ale m < = m_max
            m_values = np.arange(0.01, m_max + 0.001, 0.01)
            m = rng.choice(m_values) if len(m_values) > 0 else m_max

        # przypadek 2: d1 = 100, d2 losowane z siatki
        else:
            d1 = 100
            d2_values = np.concatenate(([0.001, 0.005], np.arange(0.01, 1.0 + 0.001, 0.01)))
            d2 = rng.choice(d2_values)

            # a z (0, 5] co 0.02
            a = rng.choice(np.arange(0.02, 5 + 0.001, 0.02))

            # m z (0,1], ale m <= a/2
            m_max = min(1.0, a / 2)
            m_values = np.arange(0.01, m_max + 0.001, 0.01)
            m = rng.choice(m_values) if len(m_values) > 0 else m_max

        dane[i] = [a, m, d1, d2]

    return dane

# ============================================================
# GENERUJ DANE
# ============================================================

print("Generuję 100,000 próbek z Twoich zakresów...")
X_new = generuj_parametry(100000)

# Przewiduj
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)

# ============================================================
# ZAPISZ
# ============================================================

df = pd.DataFrame(X_new, columns=['a', 'm', 'd1', 'd2'])
df['pattern'] = y_pred
df['pattern_name'] = [class_names[p] for p in y_pred]

for i, name in enumerate(class_names):
    df[f'prob_{name}'] = y_proba[:, i]

df.to_csv('wygenerowane_dane_100k.csv', index=False)

print(f"✅ Wygenerowano {len(df)} próbek")

# ============================================================
# POKAŻ STATYSTYKI
# ============================================================

print("\n📊 Rozkład d2:")
d2_counts = df['d2'].value_counts().head(10)
for val, count in d2_counts.items():
    print(f"   d2={val:.3f}: {count} ({count/1000:.1f}%)")

print(f"\n📊 d1=100: {sum(df['d1']==100)} próbek")
print(f"   d1≠100: {sum(df['d1']!=100)} próbek")

print("\n📊 Rozkład wzorów:")
for name in class_names:
    count = sum(df['pattern_name'] == name)
    print(f"   {name}: {count} ({count/1000:.1f}%)")


df_dupa = pd.read_csv("wygenerowane_dane_100k.csv")

df_dupa

sorted(df_dupa.iloc[:, 3].unique())

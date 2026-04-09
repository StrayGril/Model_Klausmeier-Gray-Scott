import joblib
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# WCZYTANIE ZAPISANEGO MODELU
# ============================================================

print("=== WCZYTYWANIE MODELU ===\n")

# Wczytaj model
model = joblib.load('model1_klasyfikator.pkl')
scaler = joblib.load('model1_scaler.pkl')
class_names = np.load('model1_class_names.npy', allow_pickle=True)

print(f"✅ Model wczytany pomyślnie!")
print(f"📊 Klasy: {class_names}")
print(f"🔢 Typ modelu: {type(model).__name__}")

# ============================================================
# FUNKCJA DO PRZEWIDYWANIA (POPRAWIONA!)
# ============================================================

def przewiduj_wzor(a, m, d1, d2):
    """
    Przewiduje wzór dla podanych parametrów
    """
    parametry = np.array([[a, m, d1, d2]])
    
    # Używamy surowych parametrów, tak jak podczas trenowania
    params_use = parametry
    
    probs = model.predict_proba(params_use)[0]
    pred = model.predict(params_use)[0]
    
    return class_names[pred], probs

# ============================================================
# TESTOWANIE
# ============================================================

print("\n" + "="*60)
print("TESTOWANIE MODELU")
print("="*60)

# Test 1: Według reguły 
print("\n1. Parametry zgodne z regułą:")
wzor, probs = przewiduj_wzor(2.5, 0.45, 100, 0.15)
print(f"   a=2.5, m=0.45, d1=1.5, d2=0.15")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 2: Niskie opady 
print("\n2. Niskie opady:")
wzor, probs = przewiduj_wzor(0.5, 0.5, 100, 0.78)
print(f"   a=0.5, m=0.5, d1=1.0, d2=0.1")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 3: Średnie opady, 
print("\n3. Średnie opady, niska dyfuzja biomasy (pasy):")
wzor, probs = przewiduj_wzor(1.5, 0.5, 3, 0.02)
print(f"   a=1.5, m=0.5, d1=1.0, d2=0.05")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 4: Średnie opady, wysoka dyfuzja (plamy)
print("\n4. Średnie opady, wysoka dyfuzja biomasy (plamy):")
wzor, probs = przewiduj_wzor(1.5, 0.5, 34, 0.02)
print(f"   a=1.5, m=0.5, d1=1.0, d2=0.3")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 5: Wysokie opady, niska śmiertelność (labirynt)
print("\n5. Wysokie opady, niska śmiertelność (labirynt):")
wzor, probs = przewiduj_wzor(3.0, 0.3, 15, 0.02)
print(f"   a=3.0, m=0.3, d1=1.5, d2=0.2")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# ============================================================
# WIZUALIZACJA
# ============================================================

print("\n" + "="*60)
print("WIZUALIZACJA DLA PRZYKŁADOWYCH PARAMETRÓW")
print("="*60)

a_test = 2.5
m_test = 0.45
d1_test = 1.5
d2_test = 0.15

wzor, probs = przewiduj_wzor(a_test, m_test, d1_test, d2_test)

print(f"\nParametry: a={a_test}, m={m_test}, d1={d1_test}, d2={d2_test}")
print(f"Przewidywany wzór: {wzor}")

plt.figure(figsize=(8, 4))
colors = ['red', 'blue', 'green', 'orange']
bars = plt.bar(class_names, probs, color=colors)
plt.title(f'Przewidywane prawdopodobieństwa wzorów\n(a={a_test}, m={m_test}, d1={d1_test}, d2={d2_test})')
plt.ylabel('Prawdopodobieństwo')
plt.ylim(0, 1)
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()


# Test 1: Różne opady
testy = [
    (0.5, 0.45, 30, 0.02, "NISKIE OPADY"),
    (2.5, 0.45, 30, 0.02, "ŚREDNIE OPADY"),
    (4.5, 0.45, 30, 0.02, "WYSOKIE OPADY"),
    (5.0, 0.45, 30, 0.02, "BARDZO WYSOKIE OPADY"),
]

print("\n1. TEST DLA RÓŻNYCH OPADÓW:")
for a, m, d1, d2, opis in testy:
    wzor, probs = przewiduj_wzor(a, m, d1, d2)
    print(f"\n   {opis}: a={a}, m={m}, d1={d1}, d2={d2}")
    print(f"   → {wzor}")
    for name, prob in zip(class_names, probs):
        print(f"     {name}: {prob:.3f}")

# Test 2: Różne dyfuzje
testy2 = [
    (2.5, 0.45, 100, 0.01, "DYFUZJA MIN"),
    (2.5, 0.45, 100, 0.015, "DYFUZJA ŚREDNIA"),
    (2.5, 0.45, 100, 0.02, "DYFUZJA MAX"),
]

print("\n\n2. TEST DLA RÓŻNYCH DYFUZJI BIOMASY (d2):")
for a, m, d1, d2, opis in testy2:
    wzor, probs = przewiduj_wzor(a, m, d1, d2)
    print(f"\n   {opis}: a={a}, m={m}, d1={d1}, d2={d2}")
    print(f"   → {wzor}")
    for name, prob in zip(class_names, probs):
        print(f"     {name}: {prob:.3f}")

# Test 3: Konkretne przykłady
testy3 = [
    (1.0, 0.45, 100, 0.01, "Przykład 1"),
    (3.0, 0.3, 24, 0.02, "Przykład 2"),
    (4.0, 0.2, 2, 0.02, "Przykład 3"),
    (4.2, 0.01, 6, 0.02, "Przykład 4"),
    (3.242, 1, 32, 0.02, "Przykład 5"),
    (2.265, 0.1, 24, 0.02, "Przykład 6"),
    (2.44, 0.02, 29, 0.02, "Przykład 7"),
    (1.1, 0.001, 40, 0.02, "Przykład 8")
]

print("\n\n3. KONKRETNE PRZYKŁADY:")
for a, m, d1, d2, opis in testy3:
    wzor, probs = przewiduj_wzor(a, m, d1, d2)
    print(f"\n   {opis}: a={a}, m={m}, d1={d1}, d2={d2}")
    print(f"   → {wzor}")
    for name, prob in zip(class_names, probs):
        print(f"     {name}: {prob:.3f}")

# Test 4: Losowe z danych
print("\n\n4. LOSOWE PRÓBKI Z TWOICH DANYCH:")
indeksy = np.random.choice(len(X), 5, replace=False)
for idx in indeksy:
    a, m, d1, d2 = X[idx]
    prawdziwy = class_names[y[idx]]
    wzor, probs = przewiduj_wzor(a, m, d1, d2)
    znak = "✅" if wzor == prawdziwy else "❌"
    print(f"\n   {znak} Parametry: a={a:.2f}, m={m:.2f}, d1={d1:.0f}, d2={d2:.3f}")
    print(f"      Prawdziwy: {prawdziwy} → Przewidziany: {wzor}")

# ============================================================
# GOTOWY DO URUCHOMIENIA Z TURINGIEM
# ============================================================

print("\n" + "="*60)
print("🎯 MODEL GOTOWY DO UŻYCIA!")
print("="*60)
print("\nAby przewidzieć wzór dla dowolnych parametrów, wpisz:")
print("   wzor, probs = przewiduj_wzor(a, m, d1, d2)")
print("\nPrzykład:")
print("   wzor, probs = przewiduj_wzor(2.5, 0.45, 100, 0.01)")
print("   print(wzor)  # wypisze np. 'plamy'")

print("\n✅ Test zakończony!")
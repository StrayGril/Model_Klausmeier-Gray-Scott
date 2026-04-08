import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KROK 1: TWORZENIE SYNTETYCZNEGO DATASETU (imituje Wasze dane)
# ============================================================


#def generate_synthetic_data(n_samples=1000):
    """
    Tworzy przykładowy dataset:
    X: parametry modelu [a, m, d1, d2]
    y: etykieta wzoru (0-pustynia, 1-pasy, 2-plamy, 3-labirynt)
    """
 #   np.random.seed(42)
    
    # Losuj parametry w sensownych zakresach (dostosujcie do Waszego modelu)
  #  a = np.random.uniform(0, 4, n_samples)      # opady
   # m = np.random.uniform(0.1, 1.0, n_samples)  # śmiertelność
    #d1 = np.random.uniform(0.1, 2.0, n_samples) # dyfuzja wody
    #d2 = np.random.uniform(0.01, 0.5, n_samples) # dyfuzja biomasy
    
    #X = np.column_stack([a, m, d1, d2])
    
    # Tworzymy etykiety na podstawie parametrów (to zastąpią Wasze rzeczywiste symulacje!)
    # To jest TYLKO przykład - wy musicie wczytać rzeczywiste wyniki symulacji
   # y = np.zeros(n_samples, dtype=int)
    
    #for i in range(n_samples):
        # Prosta reguła do generowania etykiet (TYLKO PRZYKŁAD!)
     #   if a[i] < 1.0:
      #      y[i] = 0  # pustynia
       # elif a[i] < 2.0:
        #    if d2[i] < 0.1:
         #       y[i] = 1  # pasy
          #  else:
           #     y[i] = 2  # plamy
        #else:
         #   if m[i] < 0.5:
          #      y[i] = 3  # labirynt
           # else:
            #    y[i] = 1  # pasy
    
    #return X, y

###
# ============================================================
# KROK 2: WCZYTANIE RZECZYWISTYCH DANYCH (to użyjecie!)
# ============================================================

def load_your_simulation_data(csv_file):
    """
    Wczytuje dane z symulacji i remapuje klasy do ciągłych wartości.
    """
    df = pd.read_csv(csv_file)
    
    # Diagnostyka
    print(f"Dostępne kolumny w CSV: {df.columns.tolist()}")
    print(f"Unikalne wartości w kolumnie 'pattern': {sorted(df['pattern'].unique())}")
    
    # Wybierz parametry
    X = df[['a', 'm', 'd1', 'd2']].values
    
    # Orginalne etykiety
    y_original = df['pattern'].values.astype(int)
    
    # REMAPOWANIE KLAS - to kluczowa zmiana!
    original_classes = np.unique(y_original)
    class_mapping = {old: new for new, old in enumerate(sorted(original_classes))}
    y = np.array([class_mapping[val] for val in y_original])
    
    # Mapowanie nazw (zachowując kolejność)
    pattern_mapping = {
        0: 'pustynia_las',
        1: 'plamy', 
        3: 'labirynty',
        4: 'dziury', 
        5: 'inne'
    }
    
    # Nazwy klas w nowej kolejności
    class_names = []
    for old_class in sorted(original_classes):
        if old_class in pattern_mapping:
            class_names.append(pattern_mapping[old_class])
        else:
            class_names.append(f'wzor_{old_class}')
    class_names = np.array(class_names)
    
    # Wyświetl informacje o mapowaniu
    print("\n--- MAPOWANIE KLAS ---")
    for old, new in class_mapping.items():
        print(f"Orginalna klasa {old} ({pattern_mapping.get(old, '?')}) -> nowa klasa {new}")
    
    print("\n--- PODSUMOWANIE DANYCH ---")
    print(f"Liczba próbek: {len(X)}")
    print(f"Liczba parametrów: {X.shape[1]}")
    print(f"Liczba klas: {len(class_names)}")
    print(f"Nowe wartości klas: {np.unique(y)}")
    
    print("\nRozkład klas:")
    for i, (old_class, name) in enumerate(zip(sorted(original_classes), class_names)):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        print(f"  Klasa {i} ({name}, org: {old_class}): {count} próbek ({percentage:.1f}%)")
    
    return X, y, class_names

# ============================================================
# KROK 3: GŁÓWNY PIPELINE MODELU 1
# ============================================================

# ============================================================
# ZMIENIONA FUNKCJA train_classification_model z obsługą BorderlineSMOTE
# ============================================================

def train_classification_model(X, y, class_names, model_type='random_forest', use_smote=False, 
                                smote_type='standard', verbose=True, scale_data=True):
    """
    Trenuje model klasyfikacji z ogromną liczbą dostępnych modeli!
    
    PARAMETRY:
    ----------
    X : numpy array - cechy (parametry)
    y : numpy array - etykiety (wzory)
    class_names : list - nazwy klas
    model_type : str - typ modelu (lista dostępnych poniżej)
    use_smote : bool - czy użyć SMOTE do balansowania
    smote_type : str - typ SMOTE ('standard', 'borderline1', 'borderline2')
    verbose : bool - czy wypisywać szczegóły
    scale_data : bool - czy skalować dane (dla modeli liniowych)
    
    DOSTĘPNE MODELE:
    ----------------
    'logistic'           - Regresja logistyczna
    'random_forest'      - Random Forest (domyślny)
    'xgboost'            - XGBoost (bardzo popularny)
    'lightgbm'           - LightGBM (szybki)
    'catboost'           - CatBoost (dobry dla kategorycznych)
    'gradient_boosting'  - Gradient Boosting
    'svm'                - Support Vector Machine
    'knn'                - K-Nearest Neighbors
    'decision_tree'      - Drzewo decyzyjne
    'naive_bayes'        - Naiwny Bayes
    'neural_network'     - Sieć neuronowa (MLP)
    'lda'                - Liniowa Analiza Dyskryminacyjna
    'qda'                - Kwadratowa Analiza Dyskryminacyjna
    'adaboost'           - AdaBoost
    'extra_trees'        - Extremely Randomized Trees
    'one_vs_rest_rf'     - OneVsRest z Random Forest
    'one_vs_rest_svm'    - OneVsRest z SVM
    """
    
    # ============================================================
    # PODZIAŁ DANYCH
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # ============================================================
    # SMOTE / BORDERLINESMOTE - BALANSOWANIE DANYCH
    # ============================================================
    if use_smote:
        if verbose:
            print(f"\n--- Stosuję {smote_type.upper()} do balansowania klas ---")
            print(f"Rozkład przed SMOTE: {Counter(y_train)}")
        
        # Wybór typu SMOTE
        if smote_type == 'standard':
            smote = SMOTE(random_state=42)
        elif smote_type == 'borderline1':
            smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif smote_type == 'borderline2':
            smote = BorderlineSMOTE(random_state=42, kind='borderline-2')
        else:
            print(f"Nieznany typ SMOTE: {smote_type}, używam standardowego")
            smote = SMOTE(random_state=42)
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        if verbose:
            print(f"Rozkład po SMOTE: {Counter(y_train_balanced)}")
            print(f"Liczba próbek przed SMOTE: {len(X_train)}")
            print(f"Liczba próbek po SMOTE: {len(X_train_balanced)}")
        
        X_train_to_use = X_train_balanced
        y_train_to_use = y_train_balanced
    else:
        X_train_to_use = X_train
        y_train_to_use = y_train
    
    # Reszta funkcji bez zmian (skalowanie, wybór modelu, trenowanie, ewaluacja)
    # ... (cała dalsza część pozostaje taka sama jak w Twoim kodzie)
    
    # ============================================================
    # SKALOWANIE (dla modeli wrażliwych na skalę)
    # ============================================================
    scaler = StandardScaler()
    
    # Modele, które NIE potrzebują skalowania
    no_scaling_models = [
        'random_forest', 'xgboost', 'lightgbm', 'catboost', 
        'gradient_boosting', 'decision_tree', 'extra_trees',
        'adaboost', 'naive_bayes'
    ]
    
    if scale_data and model_type not in no_scaling_models:
        X_train_scaled = scaler.fit_transform(X_train_to_use)
        X_test_scaled = scaler.transform(X_test)
        if verbose:
            print("\n--- Dane zostały przeskalowane ---")
    else:
        X_train_scaled = X_train_to_use
        X_test_scaled = X_test
        scaler.fit(X_train_to_use)
        if verbose and model_type in no_scaling_models:
            print("\n--- Skalowanie pominięte (model odporny na skalę) ---")
    
    # ============================================================
    # WYBÓR MODELU (TA SAMA CZĘŚĆ CO WCZEŚNIEJ)
    # ============================================================
    
    if verbose:
        print(f"\n--- Tworzę model: {model_type} ---")
    
    # ---- 1. REGRESJA LOGISTYCZNA ----
    if model_type == 'logistic':
        if use_smote:
            model = LogisticRegression(
                solver='lbfgs', max_iter=2000, random_state=42,
                class_weight=None
            )
        else:
            model = LogisticRegression(
                solver='lbfgs', max_iter=2000, random_state=42,
                class_weight='balanced'
            )
    
    # ---- 2. RANDOM FOREST ----
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
            class_weight='balanced' if not use_smote else None
        )
    
    # ---- 3. XGBOOST ----
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1
        )
    
    # ---- 4. LIGHTGBM ----
    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    # ---- 5. CATBOOST ----
    elif model_type == 'catboost':
        try:
            model = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=42, verbose=False
            )
        except:
            print("CatBoost nie jest zainstalowany. Użyj: pip install catboost")
            print("Zamiast tego używam Random Forest")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # ---- 6. GRADIENT BOOSTING ----
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
    
    # ---- 7. SVM ----
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced' if not use_smote else None,
            random_state=42
        )
    
    # ---- 8. KNN ----
    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='minkowski'
        )
    
    # ---- 9. DRZEWO DECYZYJNE ----
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced' if not use_smote else None,
            random_state=42
        )
    
    # ---- 10. NAIWNY BAYES ----
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    
    # ---- 11. SIEĆ NEURONOWA ----
    elif model_type == 'neural_network':
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu',
            solver='adam', max_iter=1000, random_state=42,
            early_stopping=True
        )
    
    # ---- 12. LDA ----
    elif model_type == 'lda':
        model = LinearDiscriminantAnalysis()
    
    # ---- 13. QDA ----
    elif model_type == 'qda':
        model = QuadraticDiscriminantAnalysis()
    
    # ---- 14. ADABOOST ----
    elif model_type == 'adaboost':
        model = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42
        )
    
    # ---- 15. EXTRA TREES ----
    elif model_type == 'extra_trees':
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
    
    # ---- 16. ONE VS REST z RANDOM FOREST ----
    elif model_type == 'one_vs_rest_rf':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = OneVsRestClassifier(base_model)
    
    # ---- 17. ONE VS REST z SVM ----
    elif model_type == 'one_vs_rest_svm':
        base_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        model = OneVsRestClassifier(base_model)
    
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")
    
    # ============================================================
    # TRENOWANIE
    # ============================================================
    if verbose:
        print(f"\n--- Trenuję model ---")
    
    try:
        model.fit(X_train_scaled, y_train_to_use)
    except Exception as e:
        print(f"Błąd podczas trenowania {model_type}: {e}")
        print("Spróbuj innego modelu lub sprawdź dane")
        return None, None, None, None, None
    
    # ============================================================
    # EWALUACJA
    # ============================================================
    y_pred = model.predict(X_test_scaled)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        y_pred_proba = None
        if verbose:
            print("Uwaga: Model nie obsługuje predict_proba")
    
    accuracy = np.mean(y_pred == y_test)
    
    if verbose:
        print(f"\n--- WYNIKI ---")
        print(f"Dokładność: {accuracy:.4f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model, scaler, X_test_scaled, y_test, y_pred_proba


# ============================================================
# NOWA FUNKCJA DO TESTOWANIA WSZYSTKICH MODELI Z RÓŻNYMI TYPAMI SMOTE
# ============================================================

def test_all_models_with_smote_types(X, y, class_names, smote_types=['standard', 'borderline1', 'borderline2']):
    """
    Testuje wszystkie modele dla różnych typów SMOTE i znajduje najlepszą kombinację
    """
    models_to_test = [
        'logistic',
        'random_forest',
        'xgboost',
        'lightgbm',
        'gradient_boosting',
        'svm',
        'knn',
        'decision_tree',
        'naive_bayes',
        'neural_network',
        'lda',
        'qda',
        'adaboost',
        'extra_trees'
    ]
    
    all_results = []
    
    print("=" * 80)
    print("TESTOWANIE WSZYSTKICH MODELI Z RÓŻNYMI TYPAMI SMOTE")
    print("=" * 80)
    
    for smote_type in smote_types:
        print("\n" + "=" * 80)
        print(f"📊 TESTOWANIE Z SMOTE TYP: {smote_type.upper()}")
        print("=" * 80)
        
        for model_type in models_to_test:
            print("\n" + "-" * 60)
            print(f"Testuję: {model_type.upper()} z {smote_type.upper()}")
            print("-" * 60)
            
            try:
                model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
                    X, y, class_names, 
                    model_type=model_type, 
                    use_smote=False,
                    smote_type=smote_type,
                    verbose=True
                )
                
                if model is not None:
                    y_pred = model.predict(X_test)
                    accuracy = np.mean(y_pred == y_test)
                    all_results.append({
                        'model': model_type,
                        'smote_type': smote_type,
                        'accuracy': accuracy,
                        'model_obj': model
                    })
            
            except Exception as e:
                print(f"Błąd dla {model_type} z {smote_type}: {e}")
                continue
    
    # Podsumowanie wszystkich wyników
    print("\n" + "=" * 80)
    print("🏆 PODSUMOWANIE WSZYSTKICH KOMBINACJI")
    print("=" * 80)
    print(f"{'Model':<20} {'SMOTE Type':<15} {'Dokładność':<10}")
    print("-" * 50)
    
    for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['model']:<20} {r['smote_type']:<15} {r['accuracy']:.4f}")
    
    # Najlepsza kombinacja
    if all_results:
        best = max(all_results, key=lambda x: x['accuracy'])
        print("\n" + "=" * 80)
        print(f"🏆 NAJLEPSZA KOMBINACJA: {best['model'].upper()} z {best['smote_type'].upper()}")
        print(f"   Dokładność: {best['accuracy']:.4f}")
        print("=" * 80)
        return best['model_obj'], best['model'], best['smote_type']
    
    return None, None, None

# ============================================================
# KROK 4: PRZYKŁAD UŻYCIA
# ============================================================

# Opcja A: Użycie syntetycznych danych (dla testu)
print("=== MODEL 1: KLASYFIKATOR PARAMETRÓW ===\n")
#X, y = load_your_simulation_data("D://Projekty//praca_licencjacka//Projekt-Formacje-roslinne-na-terenach-pustynniejacych//data//wykresy_etykiety_csv//patterns_all.csv")
#class_names = np.array(['pustynia_las', 'plamy', 'labirynt', 'inne', 'dziury'])

# Opcja B: Wczytanie rzeczywistych danych (ODKOMENTUJ)
X, y, class_names = load_your_simulation_data("D://Projekty//praca_licencjacka//Projekt-Formacje-roslinne-na-terenach-pustynniejacych//data//wykresy_etykiety_csv//patterns_all.csv")

print(f"Dane: {X.shape[0]} próbek, {X.shape[1]} parametry")
print(f"Klasy: {class_names}")

"""
# Trenuj model regresji logistycznej (Twój wybór!)
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type='logistic', use_smote = True
)

# Trenuj model random_forest (Twój wybór!)
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type='random_forest', use_smote = True
)
"""

# Opcja 3: Testuj wszystkie kombinacje (modele x typy SMOTE) - NAJLEPSZA OPCJA!
best_model, best_name, best_smote = test_all_models_with_smote_types(X, y, class_names)

# ============================================================
# TRENOWANIE FINALNEGO MODELU NA WSZYSTKICH DANYCH
# ============================================================

# Wybieramy najlepszy model (z test_all_models_with_smote_types)
best_model_name = best_name  # np. 'extra_trees'

print("\n" + "=" * 80)
print(f"🔥 TRENOWANIE FINALNEGO MODELU: {best_model_name.upper()}")
print("=" * 80)

# Stwórz model ręcznie (bez podziału danych!)
if best_model_name == 'extra_trees':
    final_model = ExtraTreesClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
elif best_model_name == 'random_forest':
    final_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
elif best_model_name == 'xgboost':
    final_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss'
    )
elif best_model_name == 'lightgbm':
    final_model = LGBMClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, verbose=-1
    )
elif best_model_name == 'knn':
    final_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
elif best_model_name == 'gradient_boosting':
    final_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
elif best_model_name == 'decision_tree':
    final_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
else:
    final_model = ExtraTreesClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)

# Trenuj na WSZYSTKICH danych - NIE MA PODZIAŁU!
print(f"\n📊 Trenuję na {len(X)} próbkach (100% danych)...")
final_model.fit(X, y)

print("✅ Finalny model wytrenowany pomyślnie!")

# Skaler (dla zgodności)
scaler = StandardScaler()
scaler.fit(X)

# To jest Wasz finalny model
model = final_model

# ============================================================
# KROK 5: POMIJAMY - NIE MA PODZIAŁU WIĘC NIE MA X_test
# ============================================================

print("\n" + "=" * 80)
print("✅ MODEL WYTRENOWANY NA WSZYSTKICH DANYCH")
print("=" * 80)
print(f"Model: {best_model_name.upper()}")
print(f"Liczba próbek treningowych: {len(X)}")
print(f"Liczba klas: {len(class_names)}")
print("\nModel gotowy do generowania danych!")

# Pomijamy wykresy i analizy które potrzebują X_test

# ============================================================
# KROK 6: ZAPIS MODELU DO UŻYCIA W MODELU 2
# ============================================================

# Zapisz model i skaler
joblib.dump(model, 'model1_klasyfikator.pkl')
joblib.dump(scaler, 'model1_scaler.pkl')
np.save('model1_class_names.npy', class_names)

print("\n=== MODEL ZAPISANY ===")
print("Pliki:")
print("  - model1_klasyfikator.pkl")
print("  - model1_scaler.pkl")
print("  - model1_class_names.npy")

# ============================================================
# KROK 7: PRZYKŁAD UŻYCIA ZAPISANEGO MODELU
# ============================================================

def predict_pattern(parameters, model, scaler, class_names):
    """
    Dla nowych parametrów przewiduj typ wzoru
    parameters: [a, m, d1, d2]
    """
    # Skaluj parametry
    params_scaled = scaler.transform([parameters])
    
    # Przewiduj prawdopodobieństwa
    probs = model.predict_proba(params_scaled)[0]
    
    return probs

# Przykład: nowe parametry do sprawdzenia
nowe_parametry = [2.5, 0.45, 1.5, 0.02]  # a, m, d1, d2
probs = predict_pattern(nowe_parametry, model, scaler, class_names)

print("\n=== PRZEWIDYWANIE DLA NOWYCH PARAMETRÓW ===")
print(f"Parametry: a={nowe_parametry[0]}, m={nowe_parametry[1]}, d1={nowe_parametry[2]}, d2={nowe_parametry[3]}")
print("Przewidywany rozkład wzorów:")
for name, prob in zip(class_names, probs):
    print(f"  {name}: {prob:.3f}")

# Wizualizacja
plt.figure(figsize=(8, 4))
plt.bar(class_names, probs, color=['red', 'blue', 'green', 'orange'])
plt.title('Przewidywane prawdopodobieństwa wzorów')
plt.ylabel('Prawdopodobieństwo')
plt.ylim(0, 1)
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()

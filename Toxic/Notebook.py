import os
import re
import warnings
import numpy as np
import pandas as pd

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

# ============================================================
# 1. STOPWORDS, STEMMER,
# ============================================================

def get_spanish_stopwords():
    """Carga stopwords de NLTK; si falla, usa una lista básica."""
    try:
        return set(stopwords.words("spanish"))
    except Exception:
        basicas = {
            "de","la","que","el","en","y","a","los","del","se","las","por","un",
            "para","con","no","una","su","al","lo","como","más","pero","sus",
            "le","ya","o","porque","cuando","muy","sin","sobre","también",
            "me","hasta","hay","donde","quien","desde","todo","nos","durante",
            "todos","uno","les","ni","contra","otros","ese","eso","ante",
            "ellos","e","esto","mí","antes","algunos","qué","unos","yo",
            "otro","otras","otra","él","tanto","esa","estos","mucho","quienes",
            "nada","muchos","cual","poco","ella","estar","estas","algunas",
            "algo","nosotros","mi","mis","tú","te","ti","tu","tus","ellas",
            "nosotras","vosostros","vosostras","os","mío","mía","míos",
            "mías","tuyo","tuya","tuyos","tuyas","suyo","suya","suyos","suyas",
            "nuestro","nuestra","nuestros","nuestras","vuestro","vuestra",
            "vuestros","vuestras","esos","esas","estoy","estás","está",
            "estamos","estáis","están","esté","estés","estemos","estéis",
            "estén","estaré","estarás","estará","estaremos","estaréis",
            "estarán"
        }
        return basicas


SPANISH_STOPWORDS = get_spanish_stopwords()
STEMMER = SnowballStemmer("spanish")

# Palabras clave/toxicas que NO queremos eliminar como stopwords
PALABRAS_IMPORTANTES = {
    "no","nunca","odio","asqueroso","puta","puto","putos","mierda",
    "matar","muerte","violencia","discriminar","racista","idiota",
    "asco","imbecil","imbécil","pendejo","marica","maricón","maricon",
    "zorra","perra"
}
SPANISH_STOPWORDS = SPANISH_STOPWORDS - PALABRAS_IMPORTANTES


def lematizar_simple(token: str) -> str:
    """
    Lematización muy simple basada en reglas.
    Sirve para mostrar el concepto (no es un lematizador completo).
    """
    # Plurales simples
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def limpiar_tweet_pln(texto: str) -> str:
    """
    Normalización + segmentación de oraciones + tokenización + lematización + stemming.
    """
    if not isinstance(texto, str):
        return ""

    # 1) Eliminar URLs
    texto = re.sub(r"https?://\S+|www\.\S+", " URL ", texto)

    # 2) Convertir menciones a token especial
    texto = re.sub(r"@\w+", " MENCION ", texto)

    # 3) Pasar hashtags a palabras
    texto = re.sub(r"#(\w+)", r" \1 ", texto)

    # 4) Normalizar carácteres
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü0-9!?\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    # 5) Segmentación de oraciones
    oraciones = re.split(r"[.!?]+", texto)

    tokens_finales = []

    for oracion in oraciones:
        oracion = oracion.strip()
        if not oracion:
            continue

        # 6) Tokenización con regex (palabras y números)
        tokens = re.findall(r"[a-záéíóúñü0-9]+", oracion)

        for tok in tokens:
            if len(tok) <= 2:
                # palabras cortas
                if tok in PALABRAS_IMPORTANTES:
                    tokens_finales.append(tok)
                continue

            if tok in SPANISH_STOPWORDS:
                continue

            # 7) Lematización
            lemma = lematizar_simple(tok)

            # 8) Stemming
            if len(lemma) > 5:
                lemma = STEMMER.stem(lemma)

            tokens_finales.append(lemma)

    if not tokens_finales:
        return ""

    return " ".join(tokens_finales)


# ============================================================
# 2. FEATURES NUMÉRICAS ESPECÍFICAS DE TOXICIDAD
# ============================================================

def extraer_features_toxicidad(textos_originales, textos_limpios):
    """
    Extrae features numéricas sencillas relacionadas con agresividad/toxicidad.
    Devuelve un np.array de shape (n_muestras, n_features).
    """
    features = []

    for orig, limpio in zip(textos_originales, textos_limpios):
        if not isinstance(orig, str):
            orig = ""
        if not isinstance(limpio, str):
            limpio = ""

        num_palabras = len(limpio.split())
        num_chars = len(limpio)

        # Puntuación agresiva
        exclam = orig.count("!")
        interrog = orig.count("?")
        puntos_suspensivos = orig.count("...")

        # Ratio de mayúsculas
        if len(orig) > 0:
            ratio_mayus = sum(1 for c in orig if c.isupper()) / len(orig)
        else:
            ratio_mayus = 0.0

        # Menciones y hashtags
        num_menciones = len(re.findall(r"@\w+", orig))
        num_hashtags = len(re.findall(r"#\w+", orig))

        # Longitud media de palabra
        palabras = limpio.split()
        avg_len = float(np.mean([len(p) for p in palabras])) if palabras else 0.0

        # Ratio de caracteres especiales
        especiales = len(re.findall(r"[^a-zA-Z0-9\s]", orig))
        ratio_especial = especiales / len(orig) if len(orig) > 0 else 0.0

        # Contador de insultos explícitos
        insultos = sum(1 for p in palabras if p in PALABRAS_IMPORTANTES)

        features.append([
            num_palabras,
            num_chars,
            exclam,
            interrog,
            puntos_suspensivos,
            ratio_mayus,
            num_menciones,
            num_hashtags,
            avg_len,
            ratio_especial,
            insultos
        ])

    return np.array(features, dtype=float)


# ============================================================
# 3. ENTRENAMIENTO DEL MODELO (BÚSQUEDA DE C PARA LinearSVC)
# ============================================================

def entrenar_modelo_mejorado(df_train: pd.DataFrame, n_splits: int = 5):
    """
    - Normalización + tokenización + lematización + stemming
    - TF-IDF de palabras (n-gramas 1-2) sobre texto limpio
    - TF-IDF de caracteres (n-gramas 3-5) sobre texto original
    - Features numéricas de toxicidad
    - Modelo: LinearSVC (uno por etiqueta HS, TR, AG)
    - Búsqueda de mejor C en un pequeño grid, optimizando F1_macro medio
    """

    print("\n" + "=" * 70)
    print(" ENTRENANDO MODELO (TUNING DE C PARA LinearSVC) ")
    print("=" * 70)

    # Copia para no modificar el DataFrame original
    df_train = df_train.copy()

    # 1) Limpieza avanzada de texto
    print("\n[1/5] Limpiando tweets...")
    df_train["tweet_limpio"] = df_train["text"].apply(limpiar_tweet_pln)
    df_train = df_train[df_train["tweet_limpio"] != ""].reset_index(drop=True)

    X_clean = df_train["tweet_limpio"]
    X_orig = df_train["text"].fillna("").astype(str)
    X_orig_lower = X_orig.str.lower()

    # Etiquetas multi-etiqueta
    y = df_train[["HS", "TR", "AG"]]

    # Distribución de clases
    print("\n Distribución de clases:")
    for col in ["HS", "TR", "AG"]:
        dist = y[col].value_counts().to_dict()
        total = len(y)
        c0 = dist.get(0, 0)
        c1 = dist.get(1, 0)
        print(f"  {col}: 0={c0} ({c0/total:5.2%}) | 1={c1} ({c1/total:5.2%})")

    # 2) TF-IDF de palabras (sobre texto limpio) y caracteres (sobre texto original)
    print("\n[2/5] Creando y ajustando vectorizadores TF-IDF...")

    # CountVectorizer solo para evidenciar Bag-of-Words / one-hot (no se usa en el modelo final,
    # pero sirve como parte de los temas vistos en clase)
    _bow = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        min_df=3
    )
    _ = _bow.fit_transform(X_clean)

    tfidf_word = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    tfidf_char = TfidfVectorizer(
        max_features=8000,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95
    )

    X_tfidf_word = tfidf_word.fit_transform(X_clean)
    X_tfidf_char = tfidf_char.fit_transform(X_orig_lower)

    print(f"  TF-IDF palabras (clean):   {X_tfidf_word.shape[1]}")
    print(f"  TF-IDF caracteres (orig):  {X_tfidf_char.shape[1]}")

    # 3) Features numéricas de toxicidad
    print("\n[3/5] Extrayendo features numéricas de toxicidad...")
    feats_tox = extraer_features_toxicidad(X_orig.values, X_clean.values)
    scaler = StandardScaler()
    feats_tox_scaled = scaler.fit_transform(feats_tox)

    # 4) Combinar todo
    print("\n[4/5] Combinando todas las representaciones...")
    X_total = hstack([
        X_tfidf_word,
        X_tfidf_char,
        csr_matrix(feats_tox_scaled)
    ])

    print(f"  Dimensión final de X: {X_total.shape}")

    # ====================================================
    # BÚSQUEDA DE MEJOR C (LinearSVC) POR F1_MACRO MEDIO
    # ====================================================
    print("\n=== BÚSQUEDA DE C PARA LinearSVC (Stratified K-Fold) ===")

    y_combined = y["HS"].astype(str) + y["TR"].astype(str) + y["AG"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    C_grid = [0.25, 0.5, 1.0, 2.0]
    resultados_C = []

    for C in C_grid:
        print(f"\nProbando C = {C} ...")
        accs = []
        f1s = []

        for fold, (idx_tr, idx_va) in enumerate(skf.split(X_total, y_combined), start=1):
            X_tr = X_total[idx_tr]
            X_va = X_total[idx_va]
            y_tr = y.iloc[idx_tr]
            y_va = y.iloc[idx_va]

            base_clf = LinearSVC(
                C=C,
                class_weight="balanced"
            )
            clf = MultiOutputClassifier(base_clf)

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_va)

            acc = accuracy_score(y_va, y_pred)
            accs.append(acc)

            f1_labels = []
            for i, col in enumerate(["HS", "TR", "AG"]):
                f1_col = f1_score(y_va.iloc[:, i], y_pred[:, i], average="macro")
                f1_labels.append(f1_col)

            f1_mean = float(np.mean(f1_labels))
            f1s.append(f1_mean)

            print(f"  Fold {fold}/{n_splits} -> Acc={acc:.4f} | F1_macro_prom={f1_mean:.4f}")

        acc_mean = float(np.mean(accs))
        f1_mean = float(np.mean(f1s))
        resultados_C.append((C, acc_mean, f1_mean))

        print(f"\n  >> RESULTADOS C={C}:")
        print(f"     Accuracy medio: {acc_mean:.4f}")
        print(f"     F1_macro medio: {f1_mean:.4f}")

    # Elegir el mejor C según F1_macro medio
    best_C, best_acc, best_f1 = max(resultados_C, key=lambda x: x[2])

    print("\n" + "=" * 70)
    print(" RESULTADOS GRID DE C (LinearSVC + TF-IDF + FEATURES) ")
    print("=" * 70)
    for C, acc_mean, f1_mean in resultados_C:
        print(f" C={C:4} -> Accuracy medio={acc_mean:.4f} | F1_macro medio={f1_mean:.4f}")
    print("-" * 70)
    print(f" >> Mejor C seleccionado: {best_C}")
    print(f"    Accuracy medio (best): {best_acc:.4f}")
    print(f"    F1_macro medio (best): {best_f1:.4f}")
    print("=" * 70)

    # ====================================================
    # 5) ENTRENAR MODELO FINAL CON EL MEJOR C
    # ====================================================
    print("\n[5/5] Entrenando modelo final con todo el train usando C óptimo...")

    base_clf_final = LinearSVC(
        C=best_C,
        class_weight="balanced"
    )
    modelo_final = MultiOutputClassifier(base_clf_final)
    modelo_final.fit(X_total, y)

    print("\nEvaluación interna sobre todo el train (solo referencia):")
    y_pred_full = modelo_final.predict(X_total)
    acc_full = accuracy_score(y, y_pred_full)
    print(f"  Accuracy exact match (train completo): {acc_full:.4f}")
    for i, col in enumerate(["HS", "TR", "AG"]):
        f1_col = f1_score(y[col], y_pred_full[:, i], average="macro")
        print(f"  {col}: F1_macro={f1_col:.4f}")

    # Devolvemos también el C elegido por transparencia
    return modelo_final, tfidf_word, tfidf_char, scaler, best_C


# ============================================================
# 4. PREDICCIÓN SOBRE TEST Y SUBMISSION
# ============================================================

def predecir_test(
    modelo,
    tfidf_word,
    tfidf_char,
    scaler,
    df_test: pd.DataFrame,
    output_file: str = "submission_mejorado.csv"
):
    print("\n" + "=" * 70)
    print(" GENERANDO PREDICCIONES PARA TEST (LinearSVC + TF-IDF) ")
    print("=" * 70)

    df_test = df_test.copy()
    df_test["tweet_limpio"] = df_test["text"].apply(limpiar_tweet_pln)

    X_clean = df_test["tweet_limpio"]
    X_orig = df_test["text"].fillna("").astype(str)
    X_orig_lower = X_orig.str.lower()

    # Vectorizaciones consistentes con el train
    X_tfidf_word = tfidf_word.transform(X_clean)
    X_tfidf_char = tfidf_char.transform(X_orig_lower)

    # Features manuales
    feats_tox = extraer_features_toxicidad(X_orig.values, X_clean.values)
    feats_tox_scaled = scaler.transform(feats_tox)

    X_total = hstack([
        X_tfidf_word,
        X_tfidf_char,
        csr_matrix(feats_tox_scaled)
    ])

    # Predicción
    y_pred = modelo.predict(X_total)

    sub = pd.DataFrame({
        "id": df_test["id"],
        "HS": y_pred[:, 0],
        "TR": y_pred[:, 1],
        "AG": y_pred[:, 2],
    })

    print("\nDistribución de predicciones:")
    total = len(sub)
    for col in ["HS", "TR", "AG"]:
        vc = sub[col].value_counts().to_dict()
        c0 = vc.get(0, 0)
        c1 = vc.get(1, 0)
        print(f"  {col}: 0={c0} ({c0/total:5.2%}) | 1={c1} ({c1/total:5.2%})")

    sub.to_csv(output_file, index=False)
    print(f"\nArchivo de submission guardado en: {output_file}")
    return sub


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" PROYECTO TOXICIDAD: MODELO CLÁSICO CON PLN + TUNING ")
    print("=" * 70)

    # Descargar stopwords al vuelo (si no están)
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        print("No se pudieron descargar las stopwords desde NLTK, usando lista básica.")

    # Rutas robustas basadas en la ubicación de este archivo
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "toxic_train.csv")
    test_path = os.path.join(base_dir, "toxic_test.csv")

    print(f"\nCargando train desde: {train_path}")
    print(f"Cargando test  desde: {test_path}")

    df_train = pd.read_csv(train_path, delimiter=";")
    df_test = pd.read_csv(test_path, delimiter=";", encoding="latin-1")

    print(f"\nTrain shape: {df_train.shape}")
    print(f"Test shape:  {df_test.shape}")

    # Entrenar modelo con búsqueda de C
    modelo, tfidf_word, tfidf_char, scaler, best_C = entrenar_modelo_mejorado(df_train, n_splits=5)
    print(f"\nC utilizado finalmente para entrenar el modelo: {best_C}")

    # Generar predicciones para test
    predecir_test(
        modelo,
        tfidf_word,
        tfidf_char,
        scaler,
        df_test,
        output_file=os.path.join(base_dir, "submission_actualizado.csv")
    )
    print("\nProceso completo")

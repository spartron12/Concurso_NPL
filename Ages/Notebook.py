# -*- coding: utf-8 -*-
"""
AGE RANGE CLASSIFIER (Tweets) — STACKING + AGE-AFFINITY FEATURES
Objetivo: superar ~57% accuracy con modelo clásico (TF-IDF + stacking).

Uso:
  python age_classifier_boosted.py \
      --train age_train.csv \
      --test age_test.csv \
      --sep ';' \
      --kbest 60000 \
      --folds 5 \
      --outfile submission_stack.csv
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# ============================================================================
# Utils básicos
# ============================================================================
def str2bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "si", "sí"}


def read_csv(path_like, sep=";"):
    """
    Intenta leer el CSV relativo al script o a la ruta absoluta que pases.
    """
    p = Path(path_like)
    if not p.exists():
        p = Path(__file__).parent.joinpath(path_like)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró: {path_like}")
    return pd.read_csv(p, delimiter=sep, encoding="utf-8")


# ============================================================================
# Limpieza y normalización básica de texto (multi-idioma ES/EN)
# ============================================================================
URL_RE = re.compile(r"https?://\S+")
MENT_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
ELONG_RE = re.compile(r"(.)\1{2,}")
EMOJI_R = re.compile(r"[😂🤣💀😭💯🔥👀✨😍😘😅😆😉😁😎🤔🤯🤮🤡🤬🥺❤️💕💔💙💚💛💜🖤🤍🤎]+")
EMOTICONS = [
    r":\)", r":-\)", r":\(", r":-\(",
    r";\)", r";-\)", r":D", r":-D", r":P", r":-P", r"<3"
]
EMOTI_RE = re.compile("|".join(EMOTICONS))

EN_CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "we've": "we have",
    "they've": "they have", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "they'd": "they would",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "they'll": "they will", "can't": "cannot", "won't": "will not",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have"
}
ES_MAP = {
    "pa'": "para", "xq": "porque", "pq": "porque", "q": "que",
    "k": "que", "tq": "te quiero", "tkm": "te quiero mucho",
    "x": "por", "d": "de", "tb": "tambien", "tmb": "tambien"
}


def expand_contractions(s: str) -> str:
    s_low = s.lower()
    for k, v in EN_CONTRACTIONS.items():
        s_low = re.sub(rf"\b{k}\b", v, s_low)
    for k, v in ES_MAP.items():
        s_low = re.sub(rf"\b{k}\b", v, s_low)
    return s_low


def normalize(text: str) -> str:
    """
    Limpieza conservadora:
      - Reemplaza URL, menciones, hashtags, emojis, emoticones.
      - Expande contracciones ES/EN.
      - Mantiene señales de elongación ("jaaa" -> "ja elong").
    """
    if not isinstance(text, str):
        text = ""
    s = text

    s = URL_RE.sub(" url ", s)
    s = MENT_RE.sub(" user ", s)
    # hashtag_palabra -> " hashtag_palabra "
    s = HASH_RE.sub(lambda m: f" hashtag_{m.group(1).lower()} ", s)

    # emojis / emoticones
    s = EMOJI_R.sub(" emoji ", s)
    s = EMOTI_RE.sub(" emoji ", s)

    # minúsculas + contracciones
    s = expand_contractions(s)

    # elongación (cooool -> coo elong)
    s = ELONG_RE.sub(r"\1\1 elong ", s)

    # solo letras, números y espacios
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_series(series: pd.Series) -> pd.Series:
    series = series.fillna("").astype(str)
    return series.apply(normalize)


# ============================================================================
# Features numéricas + afinidad de edad
# ============================================================================
YOUNG_TOKENS = [
    "lol", "lmao", "xd", "jajaja", "jeje", "jaja",
    "bro", "hermano", "amix", "bestie", "crush", "bae",
    "uni", "universidad", "tarea", "examen", "clase", "profesor",
    "profe", "weon", "weón", "pana", "parce",
    "tiktok", "insta", "instagram", "snapchat", "discord",
    "gamer", "juego", "juegos", "fortnite", "minecraft", "roblox",
    "anime", "otaku", "manga", "kpop"
]
ADULT_TOKENS = [
    "trabajo", "oficina", "jefe", "reunión", "reunion",
    "esposa", "esposo", "marido", "pareja", "novia", "novio",
    "hijo", "hija", "hijos", "familia",
    "hipoteca", "arriendo", "alquiler", "renta",
    "sueldo", "salario", "pensión", "pension",
    "impuestos", "seguro", "vacaciones", "ahorros"
]
POLITICS_TOKENS = [
    "política", "politica", "gobierno", "presidente", "congreso",
    "senado", "diputados", "elecciones", "partido", "corrupción",
    "corrupcion", "derecha", "izquierda", "protesta", "marcha"
]
CELEB_TOKENS = [
    "farándula", "farandula", "famoso", "famosa", "celebridad",
    "actor", "actriz", "cantante", "influencer", "youtuber",
    "tiktokers", "red carpet", "premios", "gala"
]
GAMING_TOKENS = [
    "juego", "juegos", "gamer", "gaming", "stream", "streamer",
    "twitch", "fortnite", "minecraft", "lol", "league of legends",
    "valorant", "csgo", "roblox"
]
WORK_TOKENS = [
    "trabajo", "empleo", "oficina", "reunión", "reunion",
    "proyecto", "empresa", "negocio", "entrevista", "currículum",
    "curriculum", "cv", "linkedin"
]
FINANCE_TOKENS = [
    "dinero", "deuda", "deudas", "ahorros", "ahorro", "tarjeta",
    "crédito", "credito", "banco", "banca", "hipoteca",
    "inversión", "inversion", "acciones", "bolsa"
]


def count_token_hits(text: str, vocab: list) -> int:
    s = text.lower()
    return sum(1 for w in vocab if w in s)


def extra_features(series: pd.Series) -> csr_matrix:
    """
    Construye features numéricas + afinidades de edad a partir del texto bruto.
    Todas las columnas son >= 0 (apto para chi2 y NB).
    """
    txt = series.fillna("").astype(str)
    low = txt.str.lower()

    df = pd.DataFrame({
        # Básicas de forma
        "num_words": txt.str.split().str.len().clip(0, 80),
        "num_chars": txt.str.len().clip(0, 300),
        "excl": txt.str.count(r"!").clip(0, 10),
        "quest": txt.str.count(r"[¿?]").clip(0, 10),
        "dots": txt.str.count(r"\.{2,}").clip(0, 5),
        "hashtags": txt.str.count(r"#").clip(0, 10),
        "mentions": txt.str.count(r"@").clip(0, 10),
        "urls": txt.str.contains("http", case=False, na=False).astype(int),
        "digits": txt.str.count(r"\d").clip(0, 15),
        "elong": txt.str.count(r"(.)\1{2,}").clip(0, 10),
        "caps_ratio": txt.apply(
            lambda x: float(sum(c.isupper() for c in x)) / max(len(x), 1)
        ).clip(0.0, 1.0),

        # Afinidades de edad / temas
        "young_score": low.apply(lambda s: count_token_hits(s, YOUNG_TOKENS)).clip(0, 15),
        "adult_score": low.apply(lambda s: count_token_hits(s, ADULT_TOKENS)).clip(0, 15),
        "politics_score": low.apply(lambda s: count_token_hits(s, POLITICS_TOKENS)).clip(0, 10),
        "celeb_score": low.apply(lambda s: count_token_hits(s, CELEB_TOKENS)).clip(0, 10),
        "gaming_score": low.apply(lambda s: count_token_hits(s, GAMING_TOKENS)).clip(0, 10),
        "work_score": low.apply(lambda s: count_token_hits(s, WORK_TOKENS)).clip(0, 10),
        "finance_score": low.apply(lambda s: count_token_hits(s, FINANCE_TOKENS)).clip(0, 10),
    })

    return csr_matrix(df.values.astype(np.float32))


# ============================================================================
# Vectorizadores TF-IDF
# ============================================================================
def build_vectorizers(fast: bool):
    if fast:
        word_max = 45000
        char_max = 20000
    else:
        word_max = 80000
        char_max = 35000

    tfidf_word = TfidfVectorizer(
        max_features=word_max,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b",
        lowercase=True,
    )

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=char_max,
        min_df=3,
        lowercase=True,
    )

    return tfidf_word, tfidf_char


# ============================================================================
# Modelos base (stacking)
# ============================================================================
def make_calibrated_lsvc(C: float = 2.5):
    svc = LinearSVC(C=C, class_weight="balanced", random_state=42)
    # Compatibilidad con distintas versiones de sklearn
    try:
        return CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=svc, method="sigmoid", cv=3)


def build_base_models():
    """
    4 modelos fuertes para texto:
      - LinearSVC calibrado
      - SGDClassifier (modified_huber)
      - LogisticRegression
      - ComplementNB
    """
    models = [
        ("lsvc", make_calibrated_lsvc(C=2.5)),
        ("sgd", SGDClassifier(
            loss="modified_huber",
            alpha=5e-5,
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
        ("lr", LogisticRegression(
            max_iter=1500,
            C=3.0,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )),
        ("nb", ComplementNB(alpha=0.2)),
    ]
    return models


# ============================================================================
# Stacking OOF
# ============================================================================
def stacking_oof(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = build_base_models()
    y = np.array(y)
    n_classes = len(np.unique(y))
    oof_meta = np.zeros((X.shape[0], n_classes * len(models)), dtype=np.float32)
    accs = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        fold_probs = []
        cl_ref = None

        print(f"\n[STACKING] Fold {fold}/{n_splits}")
        for m_idx, (name, model) in enumerate(models):
            print(f"  Entrenando base model: {name} ...")
            model.fit(Xtr, ytr)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xva)
                classes = model.classes_
            else:
                scores = model.decision_function(Xva)
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                proba = e / (e.sum(axis=1, keepdims=True) + 1e-12)
                classes = np.unique(y)

            if cl_ref is None:
                cl_ref = classes

            aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
            for ci, c in enumerate(classes):
                j = np.where(np.array(cl_ref) == c)[0]
                if len(j):
                    aligned[:, j[0]] = proba[:, ci]
            fold_probs.append(aligned)

            start = m_idx * n_classes
            end = start + n_classes
            oof_meta[va, start:end] = aligned

        # Predicción promedio entre modelos base
        stack = sum(fold_probs) / len(fold_probs)
        pred = np.array(cl_ref)[np.argmax(stack, axis=1)]
        acc = accuracy_score(yva, pred)
        accs.append(acc)
        print(f"  [OOF] Fold {fold} Accuracy: {acc:.4f}")

    print("\n[OOF] Mean ± std: {:.4f} ± {:.4f}".format(np.mean(accs), np.std(accs)))

    # Re-entrenar modelos base en TODO el train
    fitted = []
    for (name, model) in build_base_models():
        print(f"Re-entrenando modelo base completo: {name} ...")
        model.fit(X, y)
        fitted.append((name, model))

    # Meta-modelo (LogReg)
    meta = LogisticRegression(
        max_iter=800,
        C=5.0,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    print("Entrenando meta-modelo (LogisticRegression) con OOF features...")
    meta.fit(oof_meta, y)

    return fitted, meta, oof_meta, accs


def meta_transform_proba(base_models, X, n_classes, class_order):
    probs = []
    for (name, model) in base_models:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            classes = model.classes_
        else:
            scores = model.decision_function(X)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
            classes = class_order

        aligned = np.zeros((X.shape[0], n_classes), dtype=np.float32)
        for ci, c in enumerate(classes):
            j = np.where(np.array(class_order) == c)[0]
            if len(j):
                aligned[:, j[0]] = p[:, ci]
        probs.append(aligned)

    return np.hstack(probs)


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="age_train.csv", help="Ruta a age_train.csv")
    ap.add_argument("--test", default="age_test.csv", help="Ruta a age_test.csv")
    ap.add_argument("--sep", default=";", help="Delimitador CSV (';' o ',')")
    ap.add_argument("--fast", default="true", help="true/false: menos features, más rápido")
    ap.add_argument("--kbest", type=int, default=60000, help="SelectKBest(chi2) k (0=off)")
    ap.add_argument("--folds", type=int, default=5, help="Folds para OOF")
    ap.add_argument("--outfile", default="submission_stack.csv", help="Archivo de salida")
    return ap.parse_args()


def main():
    args = parse_args()
    fast = str2bool(args.fast)

    print("\n" + "=" * 80)
    print(" AGE RANGE CLASSIFIER — STACKING + AGE-AFFINITY FEATURES ")
    print("=" * 80)
    print(f"fast={fast} | kbest={args.kbest} | folds={args.folds}")
    print(f"train={args.train} | test={args.test} | sep='{args.sep}'")
    print("=" * 80 + "\n")

    # -------------------
    # Cargar TRAIN
    # -------------------
    train = read_csv(args.train, sep=args.sep)
    if not {"text", "age_range"}.issubset(train.columns):
        sys.exit("El CSV de train debe tener columnas: text, age_range")

    train = train.dropna(subset=["text", "age_range"])
    train = train[train["text"].astype(str).str.len() > 3].reset_index(drop=True)

    y = train["age_range"]
    classes = np.unique(y)
    print(f"[INFO] Clases encontradas: {classes} (n={len(classes)})")
    print(f"[INFO] Registros de train: {len(train)}")

    # -------------------
    # Limpieza de texto
    # -------------------
    print("Normalizando texto de TRAIN ...")
    text_clean = clean_series(train["text"])

    # -------------------
    # Features extra (num + afinidad edad)
    # -------------------
    print("Construyendo features numéricas + afinidad de edad (TRAIN) ...")
    feat_train = extra_features(train["text"])

    # -------------------
    # TF-IDF
    # -------------------
    print("Vectorizando TF-IDF (palabras y caracteres) ...")
    tfw, tfc = build_vectorizers(fast)
    Xw = tfw.fit_transform(text_clean)
    Xc = tfc.fit_transform(text_clean)

    X_text = hstack([Xw, Xc]).tocsr()

    # -------------------
    # Selección de características (solo sobre texto TF-IDF)
    # -------------------
    selector = None
    if args.kbest and args.kbest > 0 and args.kbest < X_text.shape[1]:
        print(f"SelectKBest(chi2) -> k={args.kbest}")
        selector = SelectKBest(chi2, k=args.kbest)
        X_text = selector.fit_transform(X_text, y)
    else:
        print("SelectKBest desactivado o k >= #features, se usa todo el TF-IDF.")

    # Combinar TF-IDF + features extra
    X = hstack([X_text, feat_train]).tocsr()
    print(f"[INFO] Shape final de X (TRAIN): {X.shape}")

    # -------------------
    # Stacking con OOF
    # -------------------
    base_models, meta, oof_meta, accs = stacking_oof(X, y, n_splits=args.folds, random_state=42)
    cv_mean = float(np.mean(accs))
    print(f"\n[OOF] CV mean accuracy: {cv_mean:.4f} -> {cv_mean * 100:.2f}%")

    # -------------------
    # INFERENCIA EN TEST
    # -------------------
    print("\n=== INFERENCIA EN TEST ===")
    test = read_csv(args.test, sep=args.sep)
    if not {"id", "text"}.issubset(test.columns):
        sys.exit("El CSV de test debe tener columnas: id, text")

    test["text"] = test["text"].fillna("")

    print("Normalizando texto de TEST ...")
    text_clean_t = clean_series(test["text"])

    print("Construyendo features numéricas + afinidad de edad (TEST) ...")
    feat_test = extra_features(test["text"])

    print("Transformando TF-IDF en TEST ...")
    Xw_t = tfw.transform(text_clean_t)
    Xc_t = tfc.transform(text_clean_t)
    X_text_t = hstack([Xw_t, Xc_t]).tocsr()

    if selector is not None:
        X_text_t = selector.transform(X_text_t)

    X_t = hstack([X_text_t, feat_test]).tocsr()
    print(f"[INFO] Shape final de X (TEST): {X_t.shape}")

    print("Generando probabilidades base models sobre TEST ...")
    meta_input = meta_transform_proba(
        base_models,
        X_t,
        n_classes=len(classes),
        class_order=classes,
    )

    print("Prediciendo age_range final con meta-modelo ...")
    final_proba = meta.predict_proba(meta_input)
    preds = classes[np.argmax(final_proba, axis=1)]

    sub = pd.DataFrame({"id": test["id"], "age_range": preds})
    out = Path(args.outfile).resolve()
    sub.to_csv(out, index=False, sep=";")

    print("\n=== RESULTADO FINAL ===")
    print(f"CV OOF accuracy: {cv_mean:.4f} ({cv_mean * 100:.2f}%)")
    print(f"Archivo de submission generado: {out}")
    print("========================\n")


if __name__ == "__main__":
    main()

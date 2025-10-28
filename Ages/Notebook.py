# ========================================
# CELDA 1: IMPORTACIONES EXTENDIDAS
# ========================================
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK
import nltk
print("Descargando recursos de NLTK...")
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, vstack

print("✓ Todas las librerías cargadas correctamente")


# ========================================
# CELDA 2: FUNCIÓN DE LIMPIEZA MEJORADA V2
# ========================================
def limpiar_tweet_mejorado_v2(tweet):
    """
    Limpieza MENOS agresiva - preserva más información
    """
    if not isinstance(tweet, str):
        return ""
    
    try:
        stop_words = set(stopwords.words("spanish"))
    except:
        stop_words = set()
    
    # Guardar el tweet original para análisis
    tweet_original = tweet
    
    # Eliminar URLs pero marcar que existían
    tiene_url = 1 if re.search(r'https?://[^\s\n\r]+', tweet) else 0
    tweet = re.sub(r'https?://[^\s\n\r]+', ' URL ', tweet)
    
    # Mantener menciones como tokens especiales
    tweet = re.sub(r'@\w+', ' MENCION ', tweet)
    
    # Mantener hashtags como tokens especiales
    tweet = re.sub(r'#(\w+)', r' HASHTAG \1 ', tweet)
    
    # Preservar emojis como tokens
    tweet = re.sub(r'[😀-🙏]', ' EMOJI_CARA ', tweet)
    tweet = re.sub(r'[❤️💕💖💗💓💞]', ' EMOJI_CORAZON ', tweet)
    
    # Mantener signos de exclamación múltiples como característica
    tweet = re.sub(r'!{2,}', ' EXCLAMACION_MULTIPLE ', tweet)
    tweet = re.sub(r'\?{2,}', ' INTERROGACION_MULTIPLE ', tweet)
    
    # Convertir números pero mantener información
    tweet = re.sub(r'\d+', ' NUMERO ', tweet)
    
    # Mantener solo letras y tokens especiales
    tweet = re.sub(r'[^a-záéíóúñA-ZÁÉÍÓÚÑ\s]', ' ', tweet)
    
    # Tokenizar
    tweet_tokens = tweet.lower().split()
    
    # Filtrar stopwords MUY común pero mantener más palabras
    stopwords_comunes = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener']
    tweet_lista = [word for word in tweet_tokens if word not in stopwords_comunes and len(word) > 1]
    
    return ' '.join(tweet_lista)

print("✓ Función de limpieza v2 definida")


# ========================================
# CELDA 3: CARGAR Y LIMPIAR DATOS
# ========================================
print("\n" + "="*60)
print("CARGANDO Y LIMPIANDO DATOS")
print("="*60)

import os
print(f"Directorio actual: {os.getcwd()}")
print(f"\nArchivos CSV disponibles:")
archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
for archivo in archivos_csv:
    print(f"  - {archivo}")

if 'age_train.csv' in archivos_csv:
    ruta_csv = 'age_train.csv'
elif os.path.exists('../age_train.csv'):
    ruta_csv = '../age_train.csv'
else:
    raise FileNotFoundError("age_train.csv no encontrado")

df_train = pd.read_csv(ruta_csv, delimiter=";")
print(f"✓ Datos originales: {len(df_train)} tweets")

df_train.dropna(subset=['text', 'age_range'], inplace=True)
print(f"✓ Después de eliminar NaN: {len(df_train)} tweets")

print("Limpiando tweets...")
df_train['tweet_limpio'] = df_train['text'].apply(limpiar_tweet_mejorado_v2)

# Filtro menos agresivo: solo 2 palabras mínimo
df_train = df_train[df_train['tweet_limpio'].str.split().str.len() >= 2]
df_train = df_train[df_train['tweet_limpio'].str.strip() != '']
print(f"✓ Después de filtrar: {len(df_train)} tweets")

print(f"\n✓ Distribución de edades:")
print(df_train['age_range'].value_counts().sort_index())


# ========================================
# CELDA 4: CARACTERÍSTICAS EXTENDIDAS
# ========================================
print("\n" + "="*60)
print("CREANDO CARACTERÍSTICAS EXTENDIDAS")
print("="*60)

def extraer_caracteristicas_avanzadas(df):
    """Extrae características más sofisticadas"""
    
    # Básicas
    df['tweet_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    df['unique_word_ratio'] = df['text'].apply(
        lambda x: len(set(str(x).lower().split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
    )
    
    # Puntuación y símbolos
    df['exclamation_count'] = df['text'].str.count('!')
    df['question_count'] = df['text'].str.count('[¿?]')
    df['dots_count'] = df['text'].str.count(r'\.')
    df['comma_count'] = df['text'].str.count(',')
    
    # Mayúsculas
    df['caps_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['caps_ratio'] = df['caps_count'] / df['tweet_length'].replace(0, 1)
    df['has_all_caps_word'] = df['text'].apply(
        lambda x: 1 if any(word.isupper() and len(word) > 1 for word in str(x).split()) else 0
    )
    
    # Emojis y símbolos
    df['emoji_count'] = df['text'].apply(
        lambda x: len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿❤️💕💖]', str(x)))
    )
    df['has_emoji'] = (df['emoji_count'] > 0).astype(int)
    
    # Hashtags y menciones
    df['hashtag_count'] = df['text'].str.count('#')
    df['mention_count'] = df['text'].str.count('@')
    df['has_url'] = df['text'].apply(lambda x: 1 if re.search(r'https?://', str(x)) else 0)
    
    # Números
    df['digit_count'] = df['text'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['has_number'] = (df['digit_count'] > 0).astype(int)
    
    # Características lingüísticas
    df['repeated_chars'] = df['text'].apply(
        lambda x: len(re.findall(r'(.)\1{2,}', str(x)))  # jajajaja, jjjjj
    )
    df['exclamation_multiple'] = df['text'].apply(
        lambda x: 1 if '!!' in str(x) or '!!!' in str(x) else 0
    )
    df['question_multiple'] = df['text'].apply(
        lambda x: 1 if '??' in str(x) or '???' in str(x) else 0
    )
    
    # Longitud de oraciones
    df['sentence_count'] = df['text'].apply(
        lambda x: len(re.split(r'[.!?]+', str(x)))
    )
    df['avg_sentence_length'] = df['tweet_length'] / df['sentence_count'].replace(0, 1)
    
    # Palabras específicas por edad (manual feature engineering)
    palabras_jovenes = ['jajaja', 'jeje', 'xd', 'omg', 'literal', 'tipo', 'wey', 'we', 'bro', 'ptm']
    palabras_mayores = ['favor', 'gracias', 'saludos', 'cordial', 'estimado', 'señor', 'usted']
    
    df['palabras_jovenes'] = df['text'].apply(
        lambda x: sum(1 for p in palabras_jovenes if p in str(x).lower())
    )
    df['palabras_mayores'] = df['text'].apply(
        lambda x: sum(1 for p in palabras_mayores if p in str(x).lower())
    )
    
    return df

df_train = extraer_caracteristicas_avanzadas(df_train)

print("✓ Características extendidas creadas (30+ features)")
print(f"Total de columnas: {len(df_train.columns)}")


# ========================================
# CELDA 5: PREPARAR DATOS
# ========================================
print("\n" + "="*60)
print("PREPARANDO DATOS PARA ENTRENAMIENTO")
print("="*60)

X_text = df_train['tweet_limpio']

# Lista completa de características numéricas
feature_cols = [
    'tweet_length', 'word_count', 'avg_word_length', 'unique_word_ratio',
    'exclamation_count', 'question_count', 'dots_count', 'comma_count',
    'caps_count', 'caps_ratio', 'has_all_caps_word',
    'emoji_count', 'has_emoji',
    'hashtag_count', 'mention_count', 'has_url',
    'digit_count', 'has_number',
    'repeated_chars', 'exclamation_multiple', 'question_multiple',
    'sentence_count', 'avg_sentence_length',
    'palabras_jovenes', 'palabras_mayores'
]

X_features = df_train[feature_cols]
y = df_train['age_range']

# Normalizar características numéricas
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)

X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
    X_text, X_features_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"✓ Datos de entrenamiento: {len(X_text_train)}")
print(f"✓ Datos de prueba: {len(X_text_test)}")
print(f"✓ Características numéricas: {len(feature_cols)}")


# ========================================
# CELDA 6: VECTORIZACIÓN
# ========================================
print("\n" + "="*60)
print("VECTORIZACIÓN MÚLTIPLE")
print("="*60)

# TF-IDF con parámetros optimizados
print("1. TF-IDF...")
vectorizer_tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\b\w+\b'
)

X_tfidf_train = vectorizer_tfidf.fit_transform(X_text_train)
X_tfidf_test = vectorizer_tfidf.transform(X_text_test)
print(f"   ✓ TF-IDF features: {X_tfidf_train.shape[1]}")

# Count Vectorizer (frecuencias absolutas)
print("2. Count Vectorizer...")
vectorizer_count = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    binary=True  # Presencia/ausencia
)

X_count_train = vectorizer_count.fit_transform(X_text_train)
X_count_test = vectorizer_count.transform(X_text_test)
print(f"   ✓ Count features: {X_count_train.shape[1]}")

# Combinar todas las características
X_train_combined = hstack([X_tfidf_train, X_count_train, X_feat_train])
X_test_combined = hstack([X_tfidf_test, X_count_test, X_feat_test])

print(f"\n✓ Features totales: {X_train_combined.shape[1]}")
print(f"  - TF-IDF: {X_tfidf_train.shape[1]}")
print(f"  - Count: {X_count_train.shape[1]}")
print(f"  - Numéricas: {X_feat_train.shape[1]}")


# ========================================
# CELDA 7: ENSEMBLE DE MODELOS
# ========================================
print("\n" + "="*60)
print("ENTRENANDO ENSEMBLE DE MODELOS")
print("="*60)

# Modelo 1: Logistic Regression
print("\n Logistic Regression")
modelo_lr = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    C=0.3,
    solver='saga',
    random_state=42,
    n_jobs=-1
)
modelo_lr.fit(X_train_combined, y_train)
y_pred_lr = modelo_lr.predict(X_test_combined)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"   ✓ Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")

# Modelo 2: Random Forest
print("\n Random Forest")
modelo_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
modelo_rf.fit(X_train_combined, y_train)
y_pred_rf = modelo_rf.predict(X_test_combined)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"   ✓ Accuracy: {acc_rf:.4f} ({acc_rf*100:.2f}%)")

# Modelo 3: Gradient Boosting
print("\n Gradient Boosting")
modelo_gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
modelo_gb.fit(X_train_combined, y_train)
y_pred_gb = modelo_gb.predict(X_test_combined)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"   ✓ Accuracy: {acc_gb:.4f} ({acc_gb*100:.2f}%)")

# Voting Classifier (Ensemble)
print("\n Voting Ensemble")
modelo_voting = VotingClassifier(
    estimators=[
        ('lr', modelo_lr),
        ('rf', modelo_rf),
        ('gb', modelo_gb)
    ],
    voting='soft',
    n_jobs=-1
)
modelo_voting.fit(X_train_combined, y_train)
y_pred_voting = modelo_voting.predict(X_test_combined)
acc_voting = accuracy_score(y_test, y_pred_voting)
print(f"   ✓ Accuracy: {acc_voting:.4f} ({acc_voting*100:.2f}%)")

# Comparación
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)
resultados = {
    'Logistic Regression': acc_lr,
    'Random Forest': acc_rf,
    'Gradient Boosting': acc_gb,
    'Voting Ensemble': acc_voting
}

for nombre, acc in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
    print(f" {nombre:25s}: {acc:.4f} ({acc*100:.2f}%)")

# Seleccionar el mejor
mejor_accuracy = max(resultados.values())
nombre_mejor = [k for k, v in resultados.items() if v == mejor_accuracy][0]

if nombre_mejor == 'Logistic Regression':
    mejor_modelo = modelo_lr
    y_pred_final = y_pred_lr
elif nombre_mejor == 'Random Forest':
    mejor_modelo = modelo_rf
    y_pred_final = y_pred_rf
elif nombre_mejor == 'Gradient Boosting':
    mejor_modelo = modelo_gb
    y_pred_final = y_pred_gb
else:
    mejor_modelo = modelo_voting
    y_pred_final = y_pred_voting

print(f"\n Mejor modelo: {nombre_mejor}")


# ========================================
# CELDA 8: EVALUACIÓN
# ========================================
print("\n" + "="*60)
print(f"EVALUACIÓN DETALLADA - {nombre_mejor}")
print("="*60)

acc_final = accuracy_score(y_test, y_pred_final)
hamming = hamming_loss(y_test, y_pred_final)

print(f"\n Accuracy: {acc_final:.4f} ({acc_final*100:.2f}%)")
print(f" Hamming Loss: {hamming:.4f}")
print(f"\n Reporte de Clasificación:")
print(classification_report(y_test, y_pred_final))

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_final)
labels_ordenados = sorted(y.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=labels_ordenados, 
            yticklabels=labels_ordenados,
            cbar_kws={'label': 'Cantidad'})
plt.title(f'Matriz de Confusión - {nombre_mejor}', fontsize=14, fontweight='bold')
plt.ylabel('Edad Real', fontsize=12)
plt.xlabel('Edad Predicha', fontsize=12)
plt.tight_layout()
plt.show()


# ========================================
# CELDA 9: PREDICCIONES PARA SUBMISSION
# ========================================
print("\n" + "="*60)
print("GENERANDO PREDICCIONES PARA SUBMISSION")
print("="*60)

df_test = pd.read_csv('age_test.csv', delimiter=';')
print(f"✓ Datos de test cargados: {len(df_test)} tweets")

df_test['text'] = df_test['text'].fillna('')
df_test['tweet_limpio'] = df_test['text'].apply(limpiar_tweet_mejorado_v2)

# Aplicar extracción de características
df_test = extraer_caracteristicas_avanzadas(df_test)

# Vectorizar
X_test_tfidf = vectorizer_tfidf.transform(df_test['tweet_limpio'])
X_test_count = vectorizer_count.transform(df_test['tweet_limpio'])
X_test_features_scaled = scaler.transform(df_test[feature_cols])

X_test_final = hstack([X_test_tfidf, X_test_count, X_test_features_scaled])

print("✓ Datos de test procesados")

# Predecir
print("Generando predicciones...")
predicciones = mejor_modelo.predict(X_test_final)

# Submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'age_range': predicciones
})

nombre_archivo = 'age_submission_optimizado_v2.csv'
submission.to_csv(nombre_archivo, index=False, sep=';')

print(f"\n ARCHIVO GUARDADO: {nombre_archivo}")
print(f" Total de predicciones: {len(submission)}")
print(f"\nDistribución de predicciones:")
print(submission['age_range'].value_counts().sort_index())


# ========================================
# GENERACIÓN DE SUBMISSION.CSV
# ========================================

# Crear submission con formato Kaggle
submission = pd.DataFrame({
    'id': df_test['id'],
    'age_range': predicciones
})

# Guardar archivo
submission.to_csv('submission.csv', index=False)

print(f"Archivo guardado: submission.csv ({len(submission)} predicciones)")
print(f"  Modelo usado: {nombre_mejor}")
print(f"  Accuracy: {acc_voting:.4f} ({acc_voting*100:.2f}%)")
print("\nDistribución de predicciones:")
print(submission['age_range'].value_counts().sort_index())

# ========================================
# CELDA 10: RESUMEN FINAL
# ========================================
print("\n" + "="*60)
print("RESUMEN FINAL V2")
print("="*60)
print(f"Mejor modelo: {nombre_mejor}")
print(f"Accuracy alcanzado: {acc_final:.4f} ({acc_final*100:.2f}%)")
print(f"Hamming Loss: {hamming:.4f}")
print(f"Features totales: {X_train_combined.shape[1]}")
print(f"   - TF-IDF: {X_tfidf_train.shape[1]}")
print(f"   - Count: {X_count_train.shape[1]}")
print(f"   - Numéricas: {len(feature_cols)}")
print(f" Archivo generado: {nombre_archivo}")
print(f"\n Mejora vs versión anterior (35.81%):")
mejora = (acc_final - 0.3581) / 0.3581 * 100
print(f"   Mejora relativa: +{mejora:.1f}%")
print(f"   Mejora absoluta: +{(acc_final - 0.3581)*100:.2f} puntos porcentuales")
print("="*60)



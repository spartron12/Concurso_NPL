### Integrantes: 
Yolanda Hernández Hernández
Juan Sebastián Rodriguez Garzon
Marc Donald Diudonne Saint Armant

# Clasificación de Discursos de Odio (HS, TR, AG)

El proyecto implementa un modelo clásico de **Procesamiento de Lenguaje Natural (PLN)** para clasificar comentarios de Twitter en tres etiquetas:

- **HS**: Hate Speech (discurso de odio)  
- **TR**: Targeted (mensaje dirigido a un grupo/persona)  
- **AG**: Aggressiveness (agresividad)

El objetivo es trabajar todo el flujo de PLN visto en clase (normalización, tokenización, lematización, stemming, n-gramas, Bag-of-Words, TF-IDF, modelos, tuning de hiperparámetros) y, al mismo tiempo, maximizar el **F1 macro** en el concurso.

---

## 1. Estructura de la carpeta

La carpeta final del proyecto queda así:

```text
Toxic/
├── Notebook.py                # Script principal de entrenamiento y predicción
├── toxic_train.csv            # Dataset de entrenamiento
├── toxic_test.csv             # Dataset de test (sin etiquetas)
├── submission_actualizado.csv # Archivo de predicciones para subir al concurso
└── README.md                  # Documentación

```
---

## 1. Requisitos

Python 3.x
* Librerías principales:
* pandas, numpy
* scikit-learn
* nltk
* scipy

--
## 3. Ejecución del modelo

En la carpeta Toxic/: python3 Notebook.py
* El script realiza automáticamente:
* Carga de toxic_train.csv y toxic_test.csv.
* Entrenamiento del modelo con validación cruzada y tuning de hiperparámetro C.
* Entrenamiento final con el mejor C.
* Predicción sobre el set de test.
* Generación del archivo submission_actualizado.csv listo para el concurso.

--
## 4. Flujo general del modelo

El flujo sigue la metodología clásica de PLN: Carga de datos, preprocesamiento y normalización del texto, segmentación de oraciones y tokenización, lematización simple y stemming, extracción de características: Bag-of-Words / One-Hot, N-gramas de palabras, N-gramas de caracteres y TF-IDF.

Se consideran los Features numéricas específicas de toxicidad, se utiliza un modelo clásico de clasificación (LinearSVC + MultiOutputClassifier) y se realiza una validación cruzada + búsqueda de hiperparámetro C (tuning) finalizando con un entrenamiento final y generación de predicciones para el concurso.

### 4.1 Lematización y Stemming

En PLN, dos personas pueden escribir la misma idea con formas diferentes de una palabra:
* “insultando”, “insultar”, “insultos”
* “discriminación”, “discriminando”, “discriminó”
* “asquerosa”, “asquerosas”, “asqueroso”

Si dejamos cada forma tal cual, el modelo cree que son palabras diferentes, es decir, aumenta la dispersión del vocabulario y reduce la capacidad del modelo de generalizar, por tal motivo nosotros buscamos la reducción a su forma singular e infinitivo usando la Lematización y El stemming reduce variaciones morfológicas (“discriminado” → “discrimin”).

Esto mejora la representación del texto, permite agrupar significados y aumenta la capacidad del modelo de detectar patrones de odio incluso si el usuario escribe variaciones creativas.

### 4.2 N-gramas + TF-IDF

Los modelos clásicos no “entienden” lenguaje: necesitan convertir texto en números mediante features. Por eso aplicamos:

✓ TF-IDF: TF-IDF sirve para resaltar palabras que aparecen mucho en un tweet,pero poco en el corpus completo. Esto ayuda a captar expresiones que son especialmente relevantes para odio o agresividad: “asco”, “asqueroso”, “zorra”, “perra”,“vete”, “maldito”, “deberían”, etc.
✓ N-gramas de palabras: Los bigramas capturan expresiones como: “puta madre”, “maldito idiota”, “asco total” que tienen un significado fuerte cuando se leen juntas. Sin bigramas, el modelo perdería el sentido porque “puta” y “madre” separados no son tan informativos como “puta madre”.
✓ N-gramas de caracteres estos nos ayudan a capturar variaciones ortográficas, muy útiles en cuentos de odio: “p3rra”, “idi0ta”, “m!erda”, “zsorraaa”

### 4.3 Features numéricas de toxicidad

Además del contenido semántico, en lenguaje tóxico existen patrones estilísticos muy fuertes, uso excesivo de mayúsculas (Gritos)

* muchos signos “!!!”
* muchos signos “???”
* longitud anormal de palabras (muuuuuuuy feooo)
* muchas menciones “@”
* muchos hashtags agresivos
* cantidad de insultos explícitos
* ratio de caracteres especiales

Estas features ayudan porque las etiquetas HS / TR / AG no solo dependen del contenido, también del tono, intensidad y estilo del mensaje.

### 4.4 Modelo SVC

En clasificación de texto tradicional: SVM lineal es el modelo más sólido para texto corto y disperso (como tweets).
Funciona bien con TF-IDF de alta dimensión y es mucho más robusto que Naive Bayes, Logistic Regression y Random Forest para este tipo de tareas.

Las ventajas que encontramos son:
* Se adapta muy bien a espacios de millones de features.
* Tolera bien datos dispersos (casi todos los vectores son ceros).
* Capta límites lineales muy finos entre lenguaje ofensivo y no ofensivo.
* Tiene un parámetro C que permite controlar la agresividad del clasificador.

### 4.5 Validación cruzada + Tuning del hiperparámetro C

Ajustamos los hiperparámetros para maximizar la métrica objetivo (F1 macro).

El SVM lineal depende mucho del parámetro C: C pequeño → modelo conservador, menos overfitting y el C alto → modelo más agresivo, detecta mejor señales pero puede sobreajustar.

1. Por eso se hace una búsqueda en grid: C_grid = [0.25, 0.5, 1.0, 2.0]
2. Y para cada C se mide: F1 macro por fold y F1 macro promedio
3. Luego se elige el C ganador, el que maximiza el F1 macro: best_C = max(resultados_C, key=lambda x: x[2])

<img width="1326" height="1430" alt="image" src="https://github.com/user-attachments/assets/167905b3-e7c6-4646-890f-b16b2bb05620" />
<img width="1466" height="1202" alt="image" src="https://github.com/user-attachments/assets/42e481e4-bc46-46d3-80f1-c033098c6753" />


### Conclusión

El modelo desarrollado alcanzó un accuracy de 0.75426, lo cual representa un desempeño sólido dentro del contexto de clasificación de discursos de odio en tweets, especialmente considerando que se trabajó con técnicas clásicas de Procesamiento de Lenguaje Natural, sin recurrir a arquitecturas profundas como transformers.

La metodología contribuyó a encontrar un balance adecuado entre generalización y sensibilidad a expresiones ofensivas:

* Lematización + stemming: reduce la variabilidad de palabras para que el modelo generalice mejor.
* TF-IDF + n-gramas: permite representar el texto con expresiones de agresión, insultos compuestos y variaciones ortográficas.
* Features numéricas: capturan estilo agresivo, gritos, repeticiones, insultos explícitos → señales muy útiles.
* LinearSVC: mejor modelo clásico para texto en alta dimensión, robusto y eficiente.
* Tuning + validación cruzada: optimiza el hiperparámetro C para maximizar F1 macro y competir mejor.

El accuracy logrado demuestra que el modelo identifica de manera efectiva las tres dimensiones del discurso tóxico (HS, TR y AG) y que las decisiones en la ingeniería de características y el ajuste de parámetros fueron apropiadas.

### Resultado Kaggle

<img width="734" height="336" alt="image" src="https://github.com/user-attachments/assets/6ecad104-083c-415d-b961-04c633fe2616" />


# Age Range Classifier — Stacking + Age-Affinity Features

Este proyecto implementa un clasificador de rangos de edad a partir de tweets, utilizando técnicas de procesamiento de lenguaje natural clásico y un esquema de stacking de modelos. Se combinan features de texto (TF-IDF) y features adicionales de afinidad de edad basadas en vocabularios temáticos.


## Estructura de la carpeta

```text
Ages/
│
├── Notebook.py
├── Age_train.csv
├── Age_test.csv
├── submission.csv

```

## Normalización y Limpieza

El texto que escriben las personas en redes sociales es altamente variado: puede ser español, inglés, spanglish o jerga. Incluye emojis, abreviaciones, errores ortográficos, hashtags, elongaciones (“holaaaaa”), sarcasmo, etc. además se mezclan temas muy diferentes según la edad (trabajo, estudios, farándula, política, gaming).

El primer paso del proyecto es identificar esa diversidad lingüística y preparar el texto para ser procesado. En ese sentido el preprocesamiento es limpiar el ruido del texto sin destruir información importante, especialmente porque señales como:

* abreviaciones,
* elongaciones,
* emojis,
* uso de jerga,
* temas recurrentes

Posteriormente el texto se procesa mediante:

- Reemplazo de URLs, menciones y hashtags.
- Expansión de contracciones en inglés y español.
- Reemplazo de emojis y emoticones por tokens genéricos.
- Detección de elongaciones de letras (cooool)
- Conservación solo de letras, números y espacios.

Expansión de contracciones que permite que los modelos capturen el significado verdadero de expresiones cortas o informales.

* En ambos idiomas (ES/EN):"can’t” → “cannot”, “i'm” → “i am”, “xq” → “porque”, “k” → “que”, “tkm” → “te quiero mucho”


##  Features adicionales

Se construyen features numéricas y de afinidad de edad:

* Básicas de forma: número de palabras, caracteres, signos de exclamación, preguntas, hashtags, menciones, URLs, dígitos, elongaciones y proporción de mayúsculas.
* Afinidad temática: presencia de tokens relacionados con:
* Jóvenes (young_score)
* Adultos (adult_score)
* Política (politics_score)
* Farándula / celebridades (celeb_score)
* Gaming (gaming_score)
* Trabajo (work_score)
* Finanzas (finance_score)

## Vectorización de texto

Se usan TF-IDF para convertir el lenguaje en números midiendo qué tan frecuente es una palabra dentro del tweet, pero qué tan rara es en el conjunto completo ayudando a identificar palabras distintivas de cada grupo de edad y patrones específicos de escritura

* Palabras (ngram 1-2)
* Caracteres (ngram 3-5)


## Modelos base

Usamos 4 modelos complementarios: LinearSVC, Logistic Regression, SGDClassifier y Naive Bayes

Cada uno captura aspectos diferentes del lenguaje:

* SVC: Bordes de decisión basados en textos largos + temas
* NB: escritura corta y ruidosa
* LR: señales suaves y probabilísticas
* SGD: variantes de texto muy disperso

Luego combinamos sus predicciones con un meta-modelo que aprende, qué modelo es mejor para cada clase y cómo ponderar sus salidas. Con esto nosotros creemos que aumentara la estabilidad y precisión.


## Stacking

El stacking de modelos clásicos (SVC, Naive Bayes, Logistic Regression y SGD) permite combinar perspectivas diferentes del texto, pero cada uno está limitado por depender exclusivamente de representaciones basadas en frecuencias. Si bien esta estrategia aumentó la robustez del modelo.

- Se realiza cross-validation OOF con StratifiedKFold.
- Se obtienen probabilidades de los modelos base.
- Se entrena un meta-modelo (LogisticRegression) sobre las probabilidades OOF.
- Finalmente, se genera la predicción combinada.

<img width="1330" height="1206" alt="image" src="https://github.com/user-attachments/assets/9245f2c1-b9c6-46b1-8499-2bbd4b484439" />


## Conclusión

El lenguaje humano es altamente complejo y ruidoso en redes sociales, especialmente en Twitter, donde aparecen abreviaciones, emojis, códigos culturales, mezcla de idiomas y jergas propias de comunidades. El modelo logró capturar parte de esa variabilidad, pero la presencia de ruido semántico todavía afecta su capacidad de generalización, lo cual se refleja en un accuracy del 45.43%.

Nosotros determinamos que la tarea a ejecutar es compleja porque la edad está solo indirectamente presente en el lenguaje, y el enfoque clásico basado en TF-IDF muestra limitaciones inherentes al no capturar el contexto semántico profundo.
Aun así, el pipeline construido y la metodológia que usamos constituye una base confiable para futuras técnicas modernas de NLP contextual y modelos preentrenados.





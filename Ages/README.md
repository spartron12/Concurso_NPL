# Age Range Classifier — Stacking + Age-Affinity Features

Este proyecto implementa un clasificador de rangos de edad a partir de tweets, utilizando técnicas de procesamiento de lenguaje natural clásico y un esquema de stacking de modelos. Se combinan features de texto (TF-IDF) y features adicionales de afinidad de edad basadas en vocabularios temáticos.

## Limpieza y normalización

El texto se procesa mediante:

- Reemplazo de URLs, menciones y hashtags.

- Expansión de contracciones en inglés y español.

- Reemplazo de emojis y emoticones por tokens genéricos.

- Detección de elongaciones de letras (cooool → coo elong).

- Conservación solo de letras, números y espacios.

##  Features adicionales

Se construyen features numéricas y de afinidad de edad:

- Básicas de forma: número de palabras, caracteres, signos de exclamación, preguntas, hashtags, menciones, URLs, dígitos, elongaciones y proporción de mayúsculas.

- Afinidad temática: presencia de tokens relacionados con:

- Jóvenes (young_score)

- Adultos (adult_score)

- Política (politics_score)

- Farándula / celebridades (celeb_score)

- Gaming (gaming_score)

- Trabajo (work_score)

- Finanzas (finance_score)

## Vectorización de texto

Se usan TF-IDF para:

- Palabras (ngram 1-2)

- Caracteres (ngram 3-5)

Opcionalmente, se puede limitar la cantidad de features para mayor velocidad (--fast true).

## Modelos base

Se entrenan cuatro modelos fuertes:

- LinearSVC calibrado (CalibratedClassifierCV)

- SGDClassifier (modified_huber)

- LogisticRegression

- ComplementNB

## Stacking

- Se realiza cross-validation OOF con StratifiedKFold.

- Se obtienen probabilidades de los modelos base.

- Se entrena un meta-modelo (LogisticRegression) sobre las probabilidades OOF.

- Finalmente, se genera la predicción combinada.

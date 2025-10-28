#Vamos a ganar!!!!!!!!!!!!

![alt text](image-1.png)

![alt text](image-2.png)


#### Deglose del código

Modelo de Machine Learning para predecir el rango de edad de usuario basándose en los textos de tweets

1. Importación de librerias
2. Limpieza de la base de datos

* Eliminar caracteres especiales, urls, arrobas, números y símpobolos especiales.
3. Tokenizan: convierte el texto minúscula y lo divide en palabras
4. Elimina Stopwords: remueve palabras comunes en español (el, la, de, etc.) que no aportan mucho significado

5. Carga y limpieza de base de datos

df_train = pd.read_csv("age_train.csv", delimiter=";")

6. Carga del dataset de entrenamiento

7. Preparación para ML

x: tweets limpios
y: rango de edad (age_range)

8. Vectorización 
9. Entrenamiento de modelo
10. Evaluación.



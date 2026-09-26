# Análisis de Reddit con NLP: de TF-IDF a LLMs

Proyecto de Procesamiento del Lenguaje Natural que compila un corpus propio de Reddit (~5.600 comentarios en 6 comunidades) y lo usa como banco de pruebas para comparar, de forma progresiva, las principales familias de técnicas de NLP: representaciones clásicas de texto, word embeddings, fine-tuning de Transformers, sentence embeddings, resumen abstractivo con LLMs y detección de contenido sensible mediante prompting.

La idea central no es solo "que funcione", sino comparar enfoques entre sí y explicar por qué uno gana a otro.

## Qué contiene

| Bloque | Objetivo | Técnicas comparadas |
| --- | --- | --- |
| Corpus | Construir y auditar un dataset propio de Reddit | Extracción desde volcados `.zst`, muestreo espaciado en el tiempo, filtrado de ruido |
| Clasificación | Predecir a qué subreddit pertenece un comentario | TF-IDF + SVM (baseline, bigramas/trigramas/char-n-gramas) · GloVe + SVM · fine-tuning de DistilBERT |
| Similitud semántica | Encontrar qué hilos se parecen entre sí | FastText (n-gramas de subpalabra) vs. sentence-transformers (`all-MiniLM-L6-v2`) |
| Resumen abstractivo | Resumir hilos completos (título + descripción + comentarios) | mT5 (`mT5_multilingual_XLSum`, fine-tuned) vs. Qwen2.5-3B-Instruct (SLM, zero-shot) |
| Moderación de contenido | Detectar comentarios sensibles en un subreddit de opinión polémica (500 comentarios) | Zero-shot, Few-shot y Chain-of-Thought prompting sobre un SLM cuantizado |

## 1. Corpus

Los datos parten de un volcado histórico de Reddit (2025) que el equipo procesó a partir de los ficheros `RS_2025.zst` / `RC_2025.zst`. Se seleccionaron 6 subreddits con vocabularios y estilos de escritura deliberadamente distintos, para que la tarea de clasificación tuviera sentido:

`r/wallstreetbets` (finanzas) · `r/travel` (viajes) · `r/buildapc` (hardware) · `r/anime` · `r/ClashRoyale` (videojuegos) · `r/relationship_advice` (relaciones)

**Criterios de muestreo**, pensados para evitar un corpus sesgado:
- Salto de 100 hilos entre cada captura en lugar de coger los primeros N, para cubrir un rango temporal amplio y no quedarse en un único pico de actividad.
- Se descartan hilos con menos de 25 comentarios y comentarios de menos de 10 palabras o que sean solo una URL.
- Separación de entrenamiento/validación **por hilo completo** (28 hilos para entrenar, 12 para validar por subreddit), para que el modelo no memorice muletillas de una conversación concreta en vez de aprender el estilo general de la comunidad.

**Resultado:** 240 hilos y 5.577 comentarios distribuidos de forma equilibrada entre los 6 subreddits (entre 857 y 979 comentarios por comunidad). El análisis exploratorio también reveló diferencias claras de estilo por comunidad — por ejemplo, los mensajes en `r/ClashRoyale` rondan los 157 caracteres de media frente a los más de 340 en `r/relationship_advice` y `r/anime` —, una señal útil de cara a la clasificación posterior.

## 2. Clasificador de subreddit

Objetivo: dado un comentario suelto, predecir en qué subreddit fue escrito. Se entrenaron y compararon tres familias de modelos sobre el mismo conjunto de validación:

| Sistema | Representación | Accuracy |
| --- | --- | --- |
| Baseline | TF-IDF + SVM lineal | 77,6 % |
| Baseline + bigramas/trigramas | TF-IDF (1,2) y (1,3) + SVM | 77,5 % / 77,2 % |
| Baseline + char-n-gramas (3-5) | TF-IDF a nivel de carácter + SVM | 77,8 % |
| Word embeddings | GloVe (Twitter, 50d) + SVM | 67,0 % |
| Transformers | Fine-tuning de DistilBERT (3 épocas) | **85 %** |

**Lectura de resultados:**
- El char-n-grama es la variante más robusta del enfoque clásico porque tolera errores ortográficos y jerga al trabajar a nivel de subpalabra, algo muy presente en Reddit.
- GloVe con vectores promediados es, paradójicamente, el peor sistema: al promediar todas las palabras de un comentario en un único vector, el significado de los términos realmente discriminantes se diluye entre el resto.
- DistilBERT es el salto de calidad real (+7,4 puntos sobre el mejor baseline clásico): al modelar la frase como secuencia y no como bolsa de palabras, capta el tono y el contexto de comunidades con vocabulario muy propio, como el argot financiero de `r/wallstreetbets`.

## 3. Similitud semántica entre hilos

Para cada hilo se concatenan todos sus comentarios y se genera un único embedding, comparando después todos los hilos entre sí con similitud coseno y visualizando el resultado como un mapa de calor.

- **FastText** (n-gramas de subpalabra): detecta algo de estructura por subreddit, pero es ruidoso — confunde hilos que comparten palabras sueltas aunque traten temas distintos.
- **Sentence-transformers** (`all-MiniLM-L6-v2`): produce bloques diagonales mucho más nítidos por subreddit y prácticamente sin falsos positivos fuera de la diagonal, gracias a que usa atención para entender la frase completa en lugar de palabras sueltas.

## 4. Resumen automático abstractivo

Cada hilo (título + descripción + comentarios) se resume con dos enfoques muy distintos:

- **mT5** (`mT5_multilingual_XLSum`), un modelo ya afinado específicamente para resumir: rápido y directo, pero al estar entrenado sobre corpus periodísticos (BBC) tiende a repetir estructuras de titular de noticia incluso cuando el texto de origen es una conversación informal de Reddit.
- **Qwen2.5-3B-Instruct**, un SLM usado en modo zero-shot: sin haber sido entrenado específicamente para resumir, produce resúmenes más naturales y captura mejor el tono conversacional y personal típico de Reddit. Requirió bajar la temperatura a 0,1 en el post-procesamiento para evitar respuestas demasiado "creativas".

Comparando 10 hilos representativos entre ambos enfoques, el SLM en zero-shot generaliza mejor a texto conversacional que un modelo especializado pero rígido — un resultado interesante sobre las ventajas del prompting frente al fine-tuning para tareas de lenguaje informal.

## 5. Detección de contenido sensible (ZSL / FSL / CoT)

Sobre un subreddit de contenido de opinión polémica en español, se extraen 500 comentarios (10 hilos × 50 comentarios) mediante muestreo aleatorio entre todos los hilos candidatos con al menos 50 comentarios — evitando así el sesgo de quedarse con los primeros hilos del fichero. Cada comentario se evalúa con un SLM cuantizado bajo tres estrategias de prompting:

- **Zero-shot (ZSL):** se le pregunta directamente si el comentario es sensible, sin ejemplos. Resultado: demasiado conservador, casi siempre responde "No" porque no tiene referencia de qué cuenta como tóxico en el registro informal de un foro, y no capta sarcasmo ni ironía.
- **Few-shot (FSL):** con un par de ejemplos de ataques personales o *ragebait*, el modelo empieza a detectar ataques sutiles que antes se le pasaban por alto.
- **Chain-of-Thought (CoT):** forzar al modelo a razonar (tema, intención, presencia de ofensa) antes de dar el veredicto final es, con diferencia, el enfoque que mejor detecta toxicidad implícita — por ejemplo, discursos agresivos sin insultos explícitos en temas de política.

**Conclusión del apartado:** para moderación de contenido en foros, las instrucciones directas rinden poco; forzar al modelo a razonar antes de decidir es lo que más se acerca a un criterio humano.

## Estructura del repositorio

```
PRACTICA2_PLNE.ipynb   Notebook completo, con celdas explicativas y resultados de cada experimento
data/
  json-principales/    Corpus crudo: 6 subreddits, hilos + comentarios sin preprocesar
  json-resumidos/      Salida del resumen abstractivo (SLM) por hilo, uno por subreddit
  resultados_contenido_sensible.json   Salida de las 3 estrategias de prompting sobre los 500 comentarios de moderación
```

## Cómo explorarlo

El notebook está pensado para ejecutarse en Google Colab con GPU (usa modelos cuantizados en 4-bit y descarga varios modelos de varios GB desde HuggingFace), y todas las celdas ya incluyen su salida — se puede leer de principio a fin sin volver a ejecutarlas. Para reproducirlo:

1. Abrir `PRACTICA2_PLNE.ipynb` en Colab.
2. Generar un token de lectura propio en [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) y añadirlo donde el notebook lo solicita.
3. Ejecutar en orden: la extracción del corpus requiere descargar los volcados `.zst` originales de Reddit, que no se incluyen en este repositorio por su tamaño.

## Limitaciones conocidas y próximas líneas

- El preprocesamiento léxico (limpieza de URLs, menciones y *stemming*) solo se probó sobre el sistema TF-IDF; comparar texto crudo vs. procesado también en GloVe y DistilBERT es el siguiente paso natural para aislar su efecto real.
- La comparación de clasificadores usa un único algoritmo (SVM) por representación; añadir Random Forest como segunda referencia, tal y como sugiere el enunciado, daría una comparación más completa.
- El resumen y la clasificación de similitud no filtran comentarios de bots de moderación (`AutoModerator`), lo que introduce algo de ruido estructurado en el corpus.
- El modelo de FastText usado es la versión más ligera (dimensión reducida); un modelo de mayor dimensión mejoraría previsiblemente la calidad de la similitud semántica.
- La comparación de LLMs para resumen y moderación se apoya en un único modelo por tarea (Qwen2.5-3B); añadir un segundo SLM permitiría diferenciar qué conclusiones dependen del modelo concreto y cuáles son generales al enfoque.

## Tecnologías

`Python` `Pandas` `Scikit-learn` `NLTK` `Gensim (GloVe)` `FastText` `PyTorch` `HuggingFace Transformers` `Sentence-Transformers` `DistilBERT` `mT5` `Qwen2.5-3B` `bitsandbytes (cuantización 4-bit)` `Google Colab`

---

## Sobre el proyecto

Desarrollado para la asignatura **Procesamiento del Lenguaje Natural (PLN)**, Grado en Ciencia e Ingeniería de Datos, Universidad de Murcia.

Proyecto en equipo de dos personas: Alejandro Gómez Javaloyes y Daniel Ruiz Gálvez.

# Laboratorio: Procesamiento de texto

Laboratorio del curso **Procesamiento de Lenguaje Natural** (Especialización en Inteligencia Artificial,
Universidad de Cundinamarca). Recorre las técnicas de procesamiento de texto, desde la normalización hasta las
tareas con modelos de lenguaje, sobre un **corpus periodístico real en español** (noticias de la agencia EFE,
CoNLL-2002). Continúa el laboratorio anterior, [`lab-embeddings`](https://github.com/Nicolas2322/lab-embeddings), y
parte del material de clase de [`lmoroney/dlaicourse`](https://github.com/lmoroney/dlaicourse).

**Autor:** Yohan Nicolás Gómez Castañeda

---

## Notebooks

| Notebook | Temas | Estado |
|---|---|---|
| `01_normalizacion_stopwords_stemming_lematizacion.ipynb` | Normalización de texto · Stopwords · Stemming · Lematización | ✅ |
| `02_…` | Term frequency · Inverse document frequency | pendiente |
| `03_…` | Part-of-speech tagging · Parsing · Named-entity recognition | pendiente |
| `04_…` | Question answering · Summarization · Sentence similarity · Text classification · Translation | pendiente |
| `05_…` | Text generation · Generación de texto con LLM | pendiente |
| `06_…` | Text mining | pendiente |

### `notebooks/01_normalizacion_stopwords_stemming_lematizacion.ipynb`

| Sección | Contenido |
|---|---|
| 1 | Corpus EFE (CoNLL-2002): reconstrucción de noticias, distribución de longitudes y ley de Zipf |
| 2 | Normalización: el `Tokenizer` de clase frente al español, etapas medidas una a una, costo de las minúsculas para las entidades, fusiones al quitar tildes, texto ruidoso de redes sociales |
| 3 | Stopwords: lista del laboratorio de clase, NLTK y spaCy; auditoría con etiquetas gramaticales de referencia; negaciones |
| 4 | Stemming con Snowball: reducción del vocabulario, sobre-agrupación y sub-agrupación |
| 5 | Lematización con spaCy: evaluación con 42 casos anotados (en contexto y aislados) y errores sobre el corpus |
| 6 | Stemming frente a lematización sobre los mismos tokens: vocabulario, legibilidad, mezcla de lemas y velocidad |
| 7 | Exportación del corpus preprocesado |

## Resultados principales (notebook 01)

| Métrica | Valor |
|---|---|
| Noticias / oraciones / tokens | 998 / 11.755 / 369.171 |
| Reducción del vocabulario por la tokenización | 29,5 % (43.553 → 30.720 tipos) |
| Reducción total de la normalización | 43,0 % |
| Tokens de entidades que se funden con palabras comunes al pasar a minúsculas | 21,6 % |
| Tokens eliminados por la lista de stopwords de clase (inglés) / NLTK / spaCy | 2,0 % / 45,7 % / 52,0 % |
| Contenido léxico entre lo eliminado: NLTK / spaCy | 1,5 % / 7,0 % |
| Reducción del vocabulario de contenido: stemming / lematización | 47,1 % / 28,0 % |
| Tokens cuya forma es una palabra real: raíces / lemas | 68,7 % / 94,8 % |
| Exactitud del lematizador en casos difíciles: en contexto / aislado | 83,3 % / 76,2 % |
| Velocidad: stemming / lematización | ≈ 3,4 veces más rápido el stemming |

## Estructura

```
lab-procesamiento-texto/
├── src/pln_lab/                 # código reutilizable por todos los notebooks
│   ├── corpus.py                # carga y reconstrucción de las noticias de CoNLL-2002
│   ├── preprocesamiento.py      # normalización, stopwords, stemming, lematización, métricas
│   └── evaluacion.py            # evaluación del lematizador contra la referencia
├── datos/
│   └── lemas_referencia.csv     # 42 casos anotados a mano para evaluar lematización
├── notebooks/
│   ├── 01_normalizacion_stopwords_stemming_lematizacion.ipynb
│   ├── figuras/                 # figuras generadas (fig01_*.png)
│   └── resultados/              # métricas en JSON (el corpus preprocesado se regenera)
├── informe/                     # informe en PDF
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Google Colab

Abrir el notebook y ejecutar todas las celdas. La primera celda detecta Colab, clona este repositorio para disponer
del paquete `pln_lab` y descarga el modelo `es_core_news_md` de spaCy. El corpus se descarga con NLTK.

### Local

```bash
git clone https://github.com/Nicolas2322/lab-procesamiento-texto.git
cd lab-procesamiento-texto
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_md
jupyter notebook notebooks/
```

Probado con Python 3.12, TensorFlow 2.21 / Keras 3.15, spaCy 3.8 y NLTK 3.10.

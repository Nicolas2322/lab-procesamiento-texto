# Laboratorio: Procesamiento de texto

Laboratorio del curso **Procesamiento de Lenguaje Natural** (Especialización en Inteligencia Artificial,
Universidad de Cundinamarca). Recorre las diecisiete técnicas de procesamiento de texto exigidas por el
laboratorio sobre un **corpus periodístico real en español** —998 noticias de la agencia EFE, del corpus
CoNLL-2002— evaluando cada técnica contra anotaciones de referencia y comparando siempre con una línea base.

Continúa el laboratorio anterior, [`lab-embeddings`](https://github.com/Nicolas2322/lab-embeddings), y parte del
material de clase de [`lmoroney/dlaicourse`](https://github.com/lmoroney/dlaicourse).

**Autor:** Yohan Nicolás Gómez Castañeda
**Docente:** Paul Alexander Díaz Montaña

---

## Mapa de los temas exigidos

| # | Tema | Notebook | Sección |
|---|---|---|---|
| 1 | Normalización de texto | `01_normalizacion_stopwords_stemming_lematizacion` | 2 |
| 2 | Stopwords | `01_normalizacion_stopwords_stemming_lematizacion` | 3 |
| 3 | Stemming | `01_normalizacion_stopwords_stemming_lematizacion` | 4 |
| 4 | Lematización | `01_normalizacion_stopwords_stemming_lematizacion` | 5 |
| 5 | Term frequency | `02_term_frequency_idf` | 2 |
| 6 | Inverse document frequency | `02_term_frequency_idf` | 3 |
| 7 | Part-of-speech (tagging) | `03_pos_parsing_ner` | 2 |
| 8 | Parsing | `03_pos_parsing_ner` | 3 |
| 9 | Named-entity recognition | `03_pos_parsing_ner` | 4 |
| 10 | Sentence similarity | `04_similitud_clasificacion` | 2 |
| 11 | Text classification | `04_similitud_clasificacion` | 3 |
| 12 | Question answering | `05_qa_resumen_traduccion` | 2 |
| 13 | Summarization | `05_qa_resumen_traduccion` | 3 |
| 14 | Translation | `05_qa_resumen_traduccion` | 4 |
| 15 | Text generation | `06_generacion_text_mining` | 2 |
| 16 | Generación de texto usando LLM | `06_generacion_text_mining` | 3 |
| 17 | Text mining | `06_generacion_text_mining` | 4 |

## Notebooks

| Notebook | Contenido |
|---|---|
| `01_normalizacion_stopwords_stemming_lematizacion.ipynb` | El `Tokenizer` de clase frente al español; etapas de normalización medidas una a una; costo de las minúsculas para las entidades; fusiones al quitar tildes; auditoría de tres listas de stopwords con las etiquetas gramaticales de referencia; Snowball frente al lematizador de spaCy evaluado con 42 casos anotados |
| `02_term_frequency_idf.ipynb` | Variantes de ponderación de TF y sesgo de longitud; idf calculado a mano y verificado contra `TfidfVectorizer`; TF-IDF, dispersión, `min_df`/`max_df` y BM25, evaluados con recuperación temática |
| `03_pos_parsing_ner.ipynb` | Etiquetado gramatical de spaCy contra las etiquetas EAGLES de referencia y contra una línea base de etiqueta más frecuente; análisis de dependencias, tripletas sujeto-verbo-objeto y ambigüedad de adjunción; NER evaluado por tramo y tipo con análisis de errores |
| `04_similitud_clasificacion.ipynb` | Similitud de oraciones con TF-IDF, LSA, vectores de spaCy, Word2Vec y SIF, evaluada por fenómeno lingüístico; clasificación temática con seis modelos, validación cruzada, curva de aprendizaje y términos informativos |
| `05_qa_resumen_traduccion.ipynb` | Sistema propio de preguntas y respuestas extractivo evaluado con las métricas de SQuAD; resumen extractivo (Lead, TextRank, LexRank, SumBasic) con métricas sin referencia; traducción basada en reglas evaluada con BLEU y chrF, con ablaciones |
| `06_generacion_text_mining.ipynb` | Generación con n-gramas y con una red `Embedding`+`Bidirectional(LSTM)`; efecto de la temperatura y del muestreo top-k; generación con LLM mediante prompting; colocaciones, modelado de temas con LDA y agrupamiento |

## Resultados principales

| Métrica | Valor |
|---|---|
| Corpus | 998 noticias · 11.755 oraciones · 369.171 tokens |
| Reducción del vocabulario por la tokenización | 29,5 % (43.553 → 30.720 tipos) |
| Tokens eliminados por las stopwords: clase (inglés) / NLTK / spaCy | 2,0 % / 45,7 % / 52,0 % |
| Exactitud del lematizador de spaCy en contexto / palabra aislada | 83,3 % / 76,2 % |
| idf calculado a mano frente a `TfidfVectorizer.idf_` | diferencia máxima 0,0 sobre 17.357 términos |
| Precisión@5 de recuperación: TF cruda / TF-IDF / TF-IDF sublineal / BM25 | 0,660 / 0,823 / 0,849 / 0,831 |
| Exactitud de POS: spaCy / línea base de etiqueta más frecuente | 85,5 % / 92,4 % (fuera de entidades: 95,2 % / 94,0 %) |
| F1 de NER por tramo y tipo (solo tramo) | 0,655 (0,846) · PER 0,835 · ORG 0,657 · LOC 0,627 · MISC 0,343 |
| Similitud de oraciones (Spearman): TF-IDF / LSA / Word2Vec+SIF / spaCy | 0,207 / 0,358 / 0,473 / 0,549 |
| Clasificación temática (5 particiones): Word2Vec+RegLog / TF-IDF+SVM / Keras | 1,000 / 0,957 / 0,957 |
| Preguntas y respuestas (37 preguntas): exact match / F1 | 59,5 / 65,0 · abstenciones correctas 4/4 |
| Traducción es→en: BLEU / chrF global (modismos) | 49,78 / 75,34 (11,59 / 52,46) |
| Perplejidad de generación: bigramas / trigramas / LSTM | 407,2 / 1.199,9 / 196,8 |
| Agrupamiento k-medias frente a los temas anotados | información mutua ajustada 0,890 · pureza 0,943 |

## Estructura

```
lab-procesamiento-texto/
├── src/pln_lab/                 # código reutilizable por todos los notebooks
│   ├── corpus.py                # carga y reconstrucción de las noticias de CoNLL-2002
│   ├── preprocesamiento.py      # normalización, stopwords, stemming, lematización, métricas
│   ├── evaluacion.py            # evaluación del lematizador contra la referencia
│   ├── ponderacion.py           # variantes de TF, idf manual y BM25
│   ├── etiquetado.py            # mapeo EAGLES→UPOS, tramos IOB y métricas de NER
│   ├── similitud.py             # representaciones de oración y correlaciones
│   ├── qa.py                    # recuperación y extracción de respuestas, métricas de SQuAD
│   ├── resumen.py               # Lead, TextRank, LexRank, SumBasic y métricas sin referencia
│   ├── traduccion.py            # traductor por diccionario y reglas
│   └── generacion.py            # n-gramas, muestreo con temperatura y top-k, diversidad
├── datos/                       # conjuntos anotados a mano para este laboratorio
│   ├── lemas_referencia.csv     # 42 casos de lematización
│   ├── sts_es.csv               # 26 pares de oraciones con similitud (0–5) por fenómeno
│   ├── temas_efe.csv            # 70 noticias etiquetadas en 5 temas
│   ├── qa_efe.csv               # 37 preguntas con respuestas literales (4 sin respuesta)
│   ├── traduccion_es_en.csv     # 16 oraciones con traducción de referencia
│   └── generacion_llm.csv       # registro de prompting con un LLM
├── notebooks/
│   ├── 01…06_*.ipynb            # los seis notebooks, ejecutados y con salidas
│   ├── figuras/                 # 42 figuras generadas
│   └── resultados/              # métricas en JSON por notebook
├── informe/
│   └── Informe_Procesamiento_de_Texto_Yohan_Nicolas_Gomez_Castañeda.pdf
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Google Colab

Abrir cualquier notebook y ejecutar todas las celdas. La primera celda detecta Colab, clona este repositorio
para disponer del paquete `pln_lab` y descarga el modelo `es_core_news_md` de spaCy; el corpus se descarga con
NLTK. Los notebooks son autocontenidos y no dependen entre sí.

### Local

```bash
git clone https://github.com/Nicolas2322/lab-procesamiento-texto.git
cd lab-procesamiento-texto
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_md
jupyter notebook notebooks/
```

Probado con Python 3.11, spaCy 3.8, NLTK 3.10, scikit-learn 1.8, gensim 4.4 y TensorFlow 2.20 (CPU).

"""Resumen extractivo de noticias y métricas para evaluarlo sin resúmenes de referencia.

Se implementan cinco selectores de oraciones, todos con la misma interfaz
(`selector(oraciones, n) -> lista de índices elegidos`), para poder compararlos sobre el
mismo corpus:

* `lead_k`: las *k* primeras oraciones. En prensa de agencia es una línea base muy fuerte
  porque la estructura de pirámide invertida coloca la información esencial al principio.
* `textrank`: grafo de oraciones con similitud del coseno TF-IDF y PageRank resuelto por
  iteración de potencias (Mihalcea y Tarau, 2004). Implementado aquí, sin librerías de grafos.
* `lexrank`: variante con grafo binarizado por umbral y coseno modificado por IDF
  (Erkan y Radev, 2004).
* `sumbasic`: selección voraz por probabilidad media de las palabras, con actualización
  a la baja de las palabras ya cubiertas (Nenkova y Vanderwende, 2005).
* `posicion_longitud`: heurística de dominio que combina la posición relativa de la
  oración con su longitud.

Las métricas sin referencia (compresión, cobertura de entidades, cobertura de términos
TF-IDF y redundancia) y un ROUGE propio (Lin, 2004) permiten evaluar sin resúmenes humanos.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_PATRON_PALABRA = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*", flags=re.UNICODE)


def fichas(texto: str, stopwords: set[str] | None = None) -> list[str]:
    tokens = [t.lower() for t in _PATRON_PALABRA.findall(texto)]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


# ---------------------------------------------------------------------------
# Representación TF-IDF interna al documento
# ---------------------------------------------------------------------------

def matriz_tfidf(oraciones: list[str], stopwords: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Matriz oraciones × vocabulario con pesos TF-IDF normalizados por filas.

    El IDF se calcula **dentro del documento** (cada oración es un documento): así el peso
    refleja qué términos distinguen una oración de las demás de la misma noticia.
    """
    bolsas = [Counter(fichas(o, stopwords)) for o in oraciones]
    vocabulario = sorted({t for b in bolsas for t in b})
    indice = {t: j for j, t in enumerate(vocabulario)}
    n = len(oraciones)
    df = Counter(t for b in bolsas for t in b)
    idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in vocabulario])
    matriz = np.zeros((n, len(vocabulario)))
    for i, bolsa in enumerate(bolsas):
        for t, c in bolsa.items():
            matriz[i, indice[t]] = (1 + math.log(c)) * idf[indice[t]]
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas, vocabulario


def similitud_coseno(matriz: np.ndarray) -> np.ndarray:
    """Matriz de similitud oración–oración (filas ya normalizadas)."""
    similitud = matriz @ matriz.T
    np.fill_diagonal(similitud, 0.0)
    return np.clip(similitud, 0.0, 1.0)


def coseno_idf(oraciones: list[str], stopwords: set[str] | None = None) -> np.ndarray:
    """Coseno modificado por IDF de LexRank: el numerador pondera cada término compartido
    por el cuadrado de su IDF."""
    bolsas = [Counter(fichas(o, stopwords)) for o in oraciones]
    n = len(oraciones)
    df = Counter(t for b in bolsas for t in b)
    idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in df}
    similitud = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            comunes = set(bolsas[i]) & set(bolsas[j])
            numerador = sum(bolsas[i][t] * bolsas[j][t] * idf[t] ** 2 for t in comunes)
            den_i = math.sqrt(sum((c * idf[t]) ** 2 for t, c in bolsas[i].items()))
            den_j = math.sqrt(sum((c * idf[t]) ** 2 for t, c in bolsas[j].items()))
            similitud[i, j] = numerador / (den_i * den_j) if den_i and den_j else 0.0
    return similitud


def pagerank(similitud: np.ndarray, amortiguacion: float = 0.85,
             tolerancia: float = 1e-6, maximo_iteraciones: int = 200) -> np.ndarray:
    """PageRank por iteración de potencias sobre el grafo de similitud (no dirigido, pesado)."""
    n = similitud.shape[0]
    if n == 0:
        return np.zeros(0)
    filas = similitud.sum(axis=1, keepdims=True)
    # Las oraciones sin ninguna similitud reparten su masa uniformemente (nodos sumidero).
    transicion = np.where(filas > 0, similitud / np.where(filas == 0, 1, filas), 1.0 / n)
    puntajes = np.full(n, 1.0 / n)
    for _ in range(maximo_iteraciones):
        nuevos = (1 - amortiguacion) / n + amortiguacion * (transicion.T @ puntajes)
        if np.abs(nuevos - puntajes).sum() < tolerancia:
            return nuevos
        puntajes = nuevos
    return puntajes


# ---------------------------------------------------------------------------
# Selectores de oraciones
# ---------------------------------------------------------------------------

def lead_k(oraciones: list[str], n: int = 3, **_) -> list[int]:
    return list(range(min(n, len(oraciones))))


def textrank(oraciones: list[str], n: int = 3, stopwords=None, **_) -> list[int]:
    matriz, _v = matriz_tfidf(oraciones, stopwords)
    puntajes = pagerank(similitud_coseno(matriz))
    elegidas = sorted(np.argsort(-puntajes)[:n])
    return [int(i) for i in elegidas]


def lexrank(oraciones: list[str], n: int = 3, stopwords=None, umbral: float = 0.1, **_) -> list[int]:
    """LexRank con grafo binarizado: se conserva la arista solo si el coseno IDF supera el
    umbral, lo que elimina las similitudes espurias por palabras comunes."""
    similitud = coseno_idf(oraciones, stopwords)
    grafo = (similitud > umbral).astype(float)
    puntajes = pagerank(grafo)
    return [int(i) for i in sorted(np.argsort(-puntajes)[:n])]


def sumbasic(oraciones: list[str], n: int = 3, stopwords=None, **_) -> list[int]:
    """SumBasic: elige la oración con mayor probabilidad media de palabra y después reduce
    al cuadrado la probabilidad de las palabras ya usadas, para evitar repetir contenido."""
    bolsas = [Counter(fichas(o, stopwords)) for o in oraciones]
    total = sum(sum(b.values()) for b in bolsas) or 1
    probabilidad = {t: c / total for t, c in Counter(t for b in bolsas for t in b.elements()).items()}
    elegidas: list[int] = []
    disponibles = set(range(len(oraciones)))
    while len(elegidas) < min(n, len(oraciones)) and disponibles:
        def media(i):
            palabras = list(bolsas[i].elements())
            return sum(probabilidad.get(t, 0.0) for t in palabras) / len(palabras) if palabras else 0.0

        mejor = max(disponibles, key=media)
        elegidas.append(mejor)
        disponibles.discard(mejor)
        for t in bolsas[mejor]:
            probabilidad[t] = probabilidad.get(t, 0.0) ** 2
    return sorted(elegidas)


def posicion_longitud(oraciones: list[str], n: int = 3, stopwords=None, **_) -> list[int]:
    """Heurística de dominio: puntaje decreciente con la posición y penalización de las
    oraciones muy cortas o muy largas respecto de la mediana del documento."""
    longitudes = np.array([len(fichas(o)) for o in oraciones], dtype=float)
    mediana = np.median(longitudes) or 1.0
    posicion = 1.0 / (1.0 + np.arange(len(oraciones)))
    penalizacion = np.exp(-np.abs(longitudes - mediana) / mediana)
    puntajes = 0.7 * posicion / posicion.max() + 0.3 * penalizacion
    return [int(i) for i in sorted(np.argsort(-puntajes)[:n])]


SELECTORES = {
    "Lead-1": lambda o, n=1, **kw: lead_k(o, 1),
    "Lead-3": lambda o, n=3, **kw: lead_k(o, 3),
    "TextRank": textrank,
    "LexRank": lexrank,
    "SumBasic": sumbasic,
    "Posición+longitud": posicion_longitud,
}


def resumir(oraciones: list[str], selector, n: int = 3, stopwords=None) -> tuple[str, list[int]]:
    indices = selector(oraciones, n=n, stopwords=stopwords)
    return " ".join(oraciones[i] for i in indices), indices


# ---------------------------------------------------------------------------
# Métricas sin referencia humana
# ---------------------------------------------------------------------------

def tasa_compresion(resumen: str, documento: str) -> float:
    """Palabras del resumen entre palabras del documento (menor = más comprimido)."""
    n_doc = len(fichas(documento))
    return len(fichas(resumen)) / n_doc if n_doc else 0.0


def cobertura_entidades(resumen: str, entidades_documento: set[str]) -> float:
    """Fracción de las entidades nombradas del documento que reaparecen en el resumen."""
    if not entidades_documento:
        return float("nan")
    texto = resumen.lower()
    presentes = sum(1 for e in entidades_documento if e.lower() in texto)
    return presentes / len(entidades_documento)


def cobertura_terminos(resumen: str, terminos_clave: list[str]) -> float:
    """Fracción de los términos con mayor TF-IDF del documento presentes en el resumen."""
    if not terminos_clave:
        return float("nan")
    presentes = set(fichas(resumen))
    return sum(1 for t in terminos_clave if t in presentes) / len(terminos_clave)


def terminos_clave(documento: str, oraciones: list[str], stopwords=None, k: int = 10) -> list[str]:
    """Los k términos con mayor peso TF-IDF acumulado dentro del documento."""
    matriz, vocabulario = matriz_tfidf(oraciones, stopwords)
    pesos = matriz.sum(axis=0)
    return [vocabulario[j] for j in np.argsort(-pesos)[:k]]


def redundancia(indices: list[int], similitud: np.ndarray) -> float:
    """Similitud media entre los pares de oraciones seleccionadas (0 si hay una sola)."""
    if len(indices) < 2:
        return 0.0
    pares = [similitud[i, j] for a, i in enumerate(indices) for j in indices[a + 1:]]
    return float(np.mean(pares)) if pares else 0.0


# ---------------------------------------------------------------------------
# ROUGE (implementación propia)
# ---------------------------------------------------------------------------

def _ngramas(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidato: str, referencia: str, n: int = 1) -> dict:
    """ROUGE-N: solapamiento de n-gramas (Lin, 2004). Devuelve precisión, recuperación y F1."""
    tc, tr = fichas(candidato), fichas(referencia)
    gc, gr = _ngramas(tc, n), _ngramas(tr, n)
    if not gc or not gr:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    comunes = sum((gc & gr).values())
    p = comunes / sum(gc.values())
    r = comunes / sum(gr.values())
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def _lcs(a: list[str], b: list[str]) -> int:
    """Longitud de la subsecuencia común más larga (programación dinámica, O(|a|·|b|))."""
    previa = [0] * (len(b) + 1)
    for x in a:
        actual = [0]
        for j, y in enumerate(b):
            actual.append(previa[j] + 1 if x == y else max(previa[j + 1], actual[j]))
        previa = actual
    return previa[-1]


def rouge_l(candidato: str, referencia: str) -> dict:
    """ROUGE-L: F1 basado en la subsecuencia común más larga."""
    tc, tr = fichas(candidato), fichas(referencia)
    if not tc or not tr:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    largo = _lcs(tc, tr)
    p, r = largo / len(tc), largo / len(tr)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def rouge_completo(candidato: str, referencia: str) -> dict:
    return {
        "ROUGE-1": rouge_n(candidato, referencia, 1)["f"],
        "ROUGE-2": rouge_n(candidato, referencia, 2)["f"],
        "ROUGE-L": rouge_l(candidato, referencia)["f"],
    }

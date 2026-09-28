"""Ponderación de términos: variantes de *term frequency*, *inverse document frequency*,
TF-IDF y BM25, más la métrica de recuperación usada para compararlas.

Las funciones trabajan sobre una matriz documento-término densa o dispersa de conteos
crudos (documentos en las filas, términos en las columnas) para que los notebooks puedan
calcular cada ponderación a mano y contrastarla con la implementación de scikit-learn.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def _a_denso(matriz) -> np.ndarray:
    return np.asarray(matriz.todense(), dtype=float) if sparse.issparse(matriz) else np.asarray(matriz, dtype=float)


# ---------------------------------------------------------------------------
# Term frequency: variantes de ponderación local
# ---------------------------------------------------------------------------

def variantes_tf(conteos) -> dict[str, np.ndarray]:
    """Devuelve las cinco ponderaciones locales clásicas (Salton y Buckley, 1988).

    - ``cruda``: el conteo tal cual; favorece los documentos largos.
    - ``binaria``: presencia o ausencia; ignora la intensidad.
    - ``relativa``: conteo dividido por la longitud del documento.
    - ``sublineal``: ``1 + ln(tf)``; la décima aparición vale menos que la segunda.
    - ``l2``: el vector de conteos escalado a norma 1 (la que aplica el coseno).
    """
    tf = _a_denso(conteos)
    longitudes = np.maximum(tf.sum(axis=1, keepdims=True), 1.0)
    return {
        "cruda": tf,
        "binaria": (tf > 0).astype(float),
        "relativa": tf / longitudes,
        "sublineal": np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1.0)), 0.0),
        "l2": normalize(tf, norm="l2"),
    }


# ---------------------------------------------------------------------------
# Inverse document frequency
# ---------------------------------------------------------------------------

def frecuencia_de_documento(conteos) -> np.ndarray:
    """df: en cuántos documentos aparece cada término al menos una vez."""
    if sparse.issparse(conteos):
        return np.asarray((conteos > 0).sum(axis=0)).ravel().astype(float)
    return (np.asarray(conteos) > 0).sum(axis=0).astype(float)


def idf_suavizado(df: np.ndarray, n_documentos: int) -> np.ndarray:
    """``idf = ln((1 + N) / (1 + df)) + 1``.

    Es la fórmula de Spärck Jones (1972) con el suavizado que usa scikit-learn
    (``smooth_idf=True``): el 1 del numerador y del denominador equivale a añadir un
    documento imaginario que contiene todos los términos (evita dividir por cero), y el
    ``+ 1`` final impide que un término presente en todos los documentos se anule.
    """
    return np.log((1.0 + n_documentos) / (1.0 + np.asarray(df, dtype=float))) + 1.0


def tf_idf(conteos, idf: np.ndarray, sublineal: bool = False, norma: str | None = "l2") -> np.ndarray:
    """Producto de la ponderación local por el idf, con normalización opcional."""
    tf = _a_denso(conteos)
    if sublineal:
        tf = np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1.0)), 0.0)
    pesos = tf * idf
    return normalize(pesos, norm=norma) if norma else pesos


# ---------------------------------------------------------------------------
# BM25 (Robertson y Zaragoza, 2009)
# ---------------------------------------------------------------------------

def idf_bm25(df: np.ndarray, n_documentos: int) -> np.ndarray:
    """idf probabilístico de BM25: ``ln(1 + (N - df + 0,5) / (df + 0,5))``."""
    df = np.asarray(df, dtype=float)
    return np.log(1.0 + (n_documentos - df + 0.5) / (df + 0.5))


def pesos_bm25(conteos, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Matriz de pesos BM25: saturación de tf controlada por ``k1`` y normalización de
    longitud controlada por ``b`` (``b = 0`` la desactiva, ``b = 1`` la aplica completa)."""
    tf = _a_denso(conteos)
    longitudes = tf.sum(axis=1)
    longitud_media = longitudes.mean() if longitudes.size else 1.0
    idf = idf_bm25(frecuencia_de_documento(tf), tf.shape[0])
    denominador = tf + k1 * (1.0 - b + b * (longitudes / longitud_media))[:, None]
    return idf * (tf * (k1 + 1.0)) / np.where(denominador == 0, 1.0, denominador)


# ---------------------------------------------------------------------------
# Evaluación: recuperación de documentos del mismo tema
# ---------------------------------------------------------------------------

def precision_en_k(pesos: np.ndarray, etiquetas: np.ndarray, k: int = 5,
                   normalizar: bool = True) -> float:
    """Precisión@k de «documento como consulta»: cada documento consulta a los demás y se
    mide qué fracción de los k primeros comparte su etiqueta. Con ``normalizar=True`` la
    similitud es el coseno; con ``False``, el producto punto (sin corregir la longitud)."""
    matriz = normalize(pesos, norm="l2") if normalizar else np.asarray(pesos, dtype=float)
    similitud = matriz @ matriz.T
    np.fill_diagonal(similitud, -np.inf)
    aciertos = [
        float(np.mean(etiquetas[np.argsort(-similitud[i])[:k]] == etiquetas[i]))
        for i in range(len(etiquetas))
    ]
    return float(np.mean(aciertos))


def precision_en_k_bm25(pesos: np.ndarray, conteos_consulta: np.ndarray,
                        etiquetas: np.ndarray, k: int = 5) -> float:
    """Precisión@k con BM25: la consulta aporta sus términos (sin peso) y el documento
    aporta el peso BM25, de modo que la puntuación no es simétrica como el coseno."""
    aciertos = []
    for i in range(len(etiquetas)):
        terminos = np.nonzero(conteos_consulta[i])[0]
        puntuacion = pesos[:, terminos].sum(axis=1)
        puntuacion[i] = -np.inf
        aciertos.append(float(np.mean(etiquetas[np.argsort(-puntuacion)[:k]] == etiquetas[i])))
    return float(np.mean(aciertos))

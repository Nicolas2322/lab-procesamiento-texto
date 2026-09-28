"""Representaciones vectoriales de oraciones y medidas de similitud.

Reúne las operaciones que comparten la sección de *sentence similarity* y la de
clasificación del notebook 04: convertir una secuencia de tokens en un único vector
(promedio simple, promedio ponderado por IDF y promedio SIF de Arora et al., 2017),
calcular el coseno por pares y evaluar la recuperación con precisión@k.

Todas las funciones trabajan sobre listas de tokens ya normalizados (lematizados y sin
stopwords), de modo que la decisión de preprocesamiento queda fuera del módulo y visible
en el notebook.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# De tokens a un vector por oración
# ---------------------------------------------------------------------------

def promedio_vectores(tokens, buscar_vector, dimension: int) -> np.ndarray:
    """Promedio simple de los vectores de los tokens que existen en el modelo.

    `buscar_vector` devuelve el vector de una palabra o None si está fuera de
    vocabulario. Si ningún token tiene vector se devuelve el vector nulo: la oración
    queda sin representación, que es exactamente lo que ocurre en la práctica con los
    métodos estáticos y conviene que se vea en las métricas.
    """
    vectores = [v for v in (buscar_vector(t) for t in tokens) if v is not None]
    if not vectores:
        return np.zeros(dimension, dtype=float)
    return np.mean(np.vstack(vectores), axis=0)


def promedio_ponderado(tokens, buscar_vector, pesos: dict, dimension: int) -> np.ndarray:
    """Promedio ponderado: cada token pesa según `pesos` (p. ej. su IDF).

    Las palabras muy frecuentes reciben poco peso, con lo que la representación deja de
    estar dominada por los términos genéricos que comparten todas las oraciones.
    """
    vectores, w = [], []
    for token in tokens:
        vector = buscar_vector(token)
        if vector is not None:
            vectores.append(vector)
            w.append(float(pesos.get(token, 1.0)))
    if not vectores or sum(w) == 0:
        return np.zeros(dimension, dtype=float)
    return np.average(np.vstack(vectores), axis=0, weights=w)


def pesos_sif(documentos_tokenizados, a: float = 1e-3) -> dict[str, float]:
    """Pesos *smooth inverse frequency*: a / (a + p(w)) con p(w) estimada en el corpus.

    Es la primera mitad del método de Arora et al. (2017). Cumple la misma función que
    el IDF —bajar el peso de lo frecuente— pero con una curva más suave y un único
    hiperparámetro.
    """
    conteo = Counter(t for documento in documentos_tokenizados for t in documento)
    total = sum(conteo.values())
    return {palabra: a / (a + c / total) for palabra, c in conteo.items()}


def quitar_componente_principal(matriz: np.ndarray, n_componentes: int = 1) -> np.ndarray:
    """Resta la proyección sobre las primeras componentes principales de la matriz.

    Segunda mitad de SIF: la dirección de mayor varianza de un conjunto de promedios
    recoge lo que todas las oraciones tienen en común (palabras de función, registro
    periodístico) y no discrimina; eliminarla suele mejorar la correlación con la
    anotación humana.
    """
    from sklearn.decomposition import TruncatedSVD

    matriz = np.asarray(matriz, dtype=float)
    if np.allclose(matriz, 0):
        return matriz
    svd = TruncatedSVD(n_components=n_componentes, random_state=0).fit(matriz)
    u = svd.components_
    return matriz - matriz.dot(u.T).dot(u)


# ---------------------------------------------------------------------------
# Similitud
# ---------------------------------------------------------------------------

def coseno_pares(A, B) -> np.ndarray:
    """Coseno fila a fila entre dos matrices con el mismo número de filas."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(axis=1)


def escalar_minmax(valores) -> np.ndarray:
    """Reescala a [0, 1] con mínimo-máximo para poder comparar perfiles entre métodos.

    Necesario porque cada representación ocupa un rango de coseno distinto (el TF-IDF
    disperso llega a 0; los promedios de embeddings raramente bajan de 0,3).
    """
    valores = np.asarray(valores, dtype=float)
    rango = valores.max() - valores.min()
    return np.zeros_like(valores) if rango == 0 else (valores - valores.min()) / rango


def precision_en_k(matriz, etiquetas, k: int = 5) -> tuple[float, np.ndarray]:
    """Precisión@k de recuperación por consulta-documento (cada fila consulta el resto).

    Devuelve la precisión media y la precisión por documento. La diagonal se excluye
    para que un documento no se recupere a sí mismo.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    etiquetas = np.asarray(etiquetas)
    similitudes = cosine_similarity(matriz)
    np.fill_diagonal(similitudes, -np.inf)
    vecinos = np.argsort(-similitudes, axis=1)[:, :k]
    por_documento = np.array([(etiquetas[vecinos[i]] == etiquetas[i]).mean() for i in range(len(etiquetas))])
    return float(por_documento.mean()), por_documento

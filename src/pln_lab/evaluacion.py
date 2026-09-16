"""Evaluación de lematizadores y stemmers contra un conjunto de referencia anotado a mano."""
from __future__ import annotations

import pandas as pd


def lematizar_en_contexto(nlp, oracion: str, palabra: str) -> str:
    """Lema que spaCy asigna a `palabra` dentro de `oracion` (primera aparición)."""
    for token in nlp(oracion):
        if token.text.lower() == palabra.lower():
            return token.lemma_.lower()
    raise ValueError(f"'{palabra}' no aparece en: {oracion}")


def lematizar_aislada(nlp, palabra: str) -> str:
    """Lema que spaCy asigna a la palabra sin contexto."""
    return nlp(palabra)[0].lemma_.lower()


def evaluar_lemas(nlp, referencia: pd.DataFrame, stemmer=None) -> pd.DataFrame:
    """Añade al conjunto de referencia el lema en contexto, el lema aislado y (opcional) la raíz."""
    resultado = referencia.copy()
    resultado["lema_contexto"] = [
        lematizar_en_contexto(nlp, o, p) for o, p in zip(referencia.oracion, referencia.palabra)
    ]
    resultado["lema_aislado"] = [lematizar_aislada(nlp, p) for p in referencia.palabra]
    resultado["acierto_contexto"] = resultado.lema_contexto == resultado.lema
    resultado["acierto_aislado"] = resultado.lema_aislado == resultado.lema
    if stemmer is not None:
        resultado["raiz"] = [stemmer(p.lower()) for p in referencia.palabra]
    return resultado

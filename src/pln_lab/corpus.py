"""Carga del corpus de trabajo: noticias de la agencia EFE (CoNLL-2002, español).

El corpus CoNLL-2002 se distribuye tokenizado, oración por oración, con etiquetas
morfosintácticas (EAGLES) y de entidades nombradas (IOB). Este módulo reconstruye los
documentos originales —cada noticia empieza con la cabecera «Ciudad, fecha (EFE)»— y
recupera un texto corrido lo más parecido posible al original, de modo que los notebooks
puedan partir de texto crudo y, a la vez, conservar las anotaciones de referencia (gold)
para evaluar etiquetadores y reconocedores en los notebooks posteriores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

ARCHIVOS_CONLL = ("esp.train", "esp.testa", "esp.testb")

# Signos que no llevan espacio antes / después al reconstruir el texto.
_SIN_ESPACIO_ANTES = set(",.;:)]}?!»%…")
_SIN_ESPACIO_DESPUES = set("([{¿¡«")


@dataclass
class Documento:
    """Una noticia con sus oraciones tokenizadas y anotadas."""

    id: int
    particion: str
    oraciones: list = field(default_factory=list)  # lista de [(palabra, pos, iob)]

    @property
    def tokens(self) -> list[str]:
        return [w for oracion in self.oraciones for w, _, _ in oracion]

    @property
    def texto(self) -> str:
        return " ".join(destokenizar([w for w, _, _ in o]) for o in self.oraciones)


def destokenizar(tokens: list[str]) -> str:
    """Reconstruye texto corrido a partir de tokens separados por espacios."""
    salida: list[str] = []
    comilla_abierta = False
    pegar_siguiente = False
    for tok in tokens:
        if tok == '"':
            if comilla_abierta:  # comilla de cierre: se pega a lo anterior
                if salida:
                    salida[-1] += tok
                else:
                    salida.append(tok)
                pegar_siguiente = False
            else:  # comilla de apertura: se pega a lo siguiente
                salida.append(tok)
                pegar_siguiente = True
            comilla_abierta = not comilla_abierta
            continue
        if salida and (pegar_siguiente or (tok and tok[0] in _SIN_ESPACIO_ANTES)):
            salida[-1] += tok
        else:
            salida.append(tok)
        pegar_siguiente = tok in _SIN_ESPACIO_DESPUES
    return " ".join(salida)


def _es_cabecera_efe(oracion) -> bool:
    toks = [w for w, _, _ in oracion[:15]]
    return "(" in toks[:10] and any(t.startswith("EFE") for t in toks)


def cargar_documentos_efe(descargar: bool = True) -> list[Documento]:
    """Devuelve las noticias de CoNLL-2002 (español) agrupadas por documento."""
    import nltk

    if descargar:
        nltk.download("conll2002", quiet=True)
    from nltk.corpus import conll2002

    documentos: list[Documento] = []
    actual: Documento | None = None
    for archivo in ARCHIVOS_CONLL:
        for oracion in conll2002.iob_sents(archivo):
            if not oracion:
                continue
            if actual is None or (_es_cabecera_efe(oracion) and actual.oraciones):
                actual = Documento(id=len(documentos), particion=archivo)
                documentos.append(actual)
            actual.oraciones.append(oracion)
    return documentos


def documentos_a_dataframe(documentos: list[Documento]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [d.id for d in documentos],
            "particion": [d.particion for d in documentos],
            "n_oraciones": [len(d.oraciones) for d in documentos],
            "n_tokens": [len(d.tokens) for d in documentos],
            "texto": [d.texto for d in documentos],
        }
    )

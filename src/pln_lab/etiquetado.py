"""Etiquetado gramatical, análisis de dependencias y entidades nombradas.

Reúne lo que el notebook 03 necesita para comparar la salida de spaCy con la anotación
humana de CoNLL-2002:

* la correspondencia entre el juego de etiquetas **EAGLES** del corpus y el juego
  **universal (UPOS)** que produce spaCy (`a_upos`, `MAPA_EAGLES`);
* la extracción de tramos de entidad a partir de etiquetas **IOB** (`tramos_iob`);
* el emparejamiento de entidades predichas y de referencia por coincidencia exacta de
  tramo y tipo, con las cuentas de verdaderos positivos, falsos positivos y falsos
  negativos (`contar_entidades`, `prf`).

No se usa ninguna librería externa de evaluación de secuencias: el emparejamiento se
implementa aquí para que el criterio quede explícito y auditable.
"""
from __future__ import annotations

from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------
# EAGLES -> UPOS
# --------------------------------------------------------------------------------------
# Las etiquetas EAGLES de CoNLL-2002 son posicionales: la primera letra indica la
# categoría y las siguientes el tipo y los rasgos morfológicos (VMI = Verbo Main
# Indicativo). El mapeo se hace por prefijo de dos letras, con dos excepciones:
# la puntuación (cualquier etiqueta que empiece por F) y las etiquetas de una letra.
MAPA_EAGLES = {
    # Nombres
    "NC": "NOUN",   # nombre común
    "NP": "PROPN",  # nombre propio
    # Verbos: el segundo carácter distingue main / auxiliar (haber) / semiauxiliar (ser)
    "VM": "VERB",
    "VA": "AUX",
    "VS": "AUX",
    # Adjetivos
    "AQ": "ADJ",    # calificativo
    "AO": "ADJ",    # ordinal
    # Adverbios
    "RG": "ADV",    # general
    "RN": "ADV",    # de negación (UPOS no tiene categoría propia)
    # Preposiciones
    "SP": "ADP",
    # Determinantes
    "DA": "DET",    # artículo
    "DD": "DET",    # demostrativo
    "DP": "DET",    # posesivo
    "DI": "DET",    # indefinido
    "DT": "DET",    # interrogativo
    "DE": "DET",    # exclamativo
    "DN": "NUM",    # numeral: UD trata los cardinales como NUM aunque actúen de determinante
    # Pronombres
    "PP": "PRON",   # personal
    "PR": "PRON",   # relativo
    "PT": "PRON",   # interrogativo
    "P0": "PRON",   # clítico reflexivo / impersonal («se»)
    "PI": "PRON",   # indefinido
    "PD": "PRON",   # demostrativo
    "PX": "PRON",   # posesivo
    "PN": "PRON",   # numeral («ciento», «ambos»)
    # Conjunciones
    "CC": "CCONJ",
    "CS": "SCONJ",
    # Otras
    "Z": "NUM",     # cifra
    "I": "INTJ",    # interjección
    "Y": "PROPN",   # abreviatura / sigla
}

CATEGORIAS_UPOS = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "ADP", "DET",
    "PRON", "NUM", "CCONJ", "SCONJ", "PUNCT", "INTJ",
]

TIPOS_ENTIDAD = ["PER", "ORG", "LOC", "MISC"]


def a_upos(etiqueta: str) -> str | None:
    """Traduce una etiqueta EAGLES del corpus al juego universal (UPOS).

    Devuelve `None` si la etiqueta no está cubierta, para poder contabilizar las
    omisiones del mapeo en lugar de ocultarlas con una categoría por defecto.
    """
    if etiqueta.startswith("F"):  # Fc, Fp, Fe, Fpa… toda la puntuación
        return "PUNCT"
    for n in (2, 1, 3):
        if etiqueta[:n] in MAPA_EAGLES:
            return MAPA_EAGLES[etiqueta[:n]]
    return None


# --------------------------------------------------------------------------------------
# Entidades: tramos IOB y emparejamiento
# --------------------------------------------------------------------------------------
def tramos_iob(etiquetas) -> list[tuple[str, int, int]]:
    """Convierte una secuencia IOB en tramos `(tipo, inicio, fin)`, con `fin` exclusivo.

    Se aplican dos reglas de tolerancia frecuentes en anotación real: una `I-X` que
    aparece sin `B-X` previa abre un tramo nuevo, y un cambio de tipo dentro de una
    secuencia `I-` cierra el tramo anterior.
    """
    tramos: list[tuple[str, int, int]] = []
    tipo: str | None = None
    inicio = 0
    for k, etiqueta in enumerate(etiquetas):
        if etiqueta.startswith("B-"):
            if tipo is not None:
                tramos.append((tipo, inicio, k))
            tipo, inicio = etiqueta[2:], k
        elif etiqueta.startswith("I-"):
            if tipo is None:
                tipo, inicio = etiqueta[2:], k
            elif etiqueta[2:] != tipo:
                tramos.append((tipo, inicio, k))
                tipo, inicio = etiqueta[2:], k
        else:  # "O"
            if tipo is not None:
                tramos.append((tipo, inicio, k))
                tipo = None
    if tipo is not None:
        tramos.append((tipo, inicio, len(etiquetas)))
    return tramos


def contar_entidades(oro, prediccion, tipos=TIPOS_ENTIDAD):
    """Verdaderos positivos, falsos positivos y falsos negativos por tipo de entidad.

    `oro` y `prediccion` son listas de listas de tramos `(tipo, inicio, fin)`, una por
    oración. Un acierto exige **coincidencia exacta de tramo y de tipo**: es el criterio
    de la tarea compartida de CoNLL-2002 y el más exigente de los habituales.
    """
    vp, fp, fn = Counter(), Counter(), Counter()
    for tramos_oro, tramos_pred in zip(oro, prediccion):
        conjunto_oro = {t for t in tramos_oro if t[0] in tipos}
        conjunto_pred = {t for t in tramos_pred if t[0] in tipos}
        for t in conjunto_oro & conjunto_pred:
            vp[t[0]] += 1
        for t in conjunto_pred - conjunto_oro:
            fp[t[0]] += 1
        for t in conjunto_oro - conjunto_pred:
            fn[t[0]] += 1
    return vp, fp, fn


def prf(vp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precisión, exhaustividad y F1 a partir de las tres cuentas."""
    precision = vp / (vp + fp) if vp + fp else 0.0
    exhaustividad = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * exhaustividad / (precision + exhaustividad) if precision + exhaustividad else 0.0
    return precision, exhaustividad, f1


def clasificar_error(tramo_oro, tramos_pred):
    """Clasifica un falso negativo comparándolo con los tramos predichos que lo solapan.

    Devuelve `(clase, tramo_predicho_o_None)` con clase en {«tipo», «frontera», «omisión»}:
    el tramo coincide y el tipo no («tipo»), hay solapamiento parcial («frontera») o no
    hay ninguna predicción sobre esas palabras («omisión»).
    """
    _, inicio, fin = tramo_oro
    solapan = sorted(t for t in tramos_pred if t[1] < fin and t[2] > inicio)
    if not solapan:
        return "omisión", None
    candidato = solapan[0]
    if (candidato[1], candidato[2]) == (inicio, fin):
        return "tipo", candidato
    return "frontera", candidato


# --------------------------------------------------------------------------------------
# Línea base de etiquetado: etiqueta más frecuente por palabra
# --------------------------------------------------------------------------------------
class EtiquetadorMasFrecuente:
    """Asigna a cada palabra la categoría con la que aparece más veces en el entrenamiento.

    Es la línea base clásica del etiquetado gramatical (Jurafsky y Martin, 2025): no usa
    contexto, solo memoriza. Las palabras no vistas reciben la categoría mayoritaria del
    corpus de entrenamiento.
    """

    def __init__(self):
        self.tabla: dict[str, str] = {}
        self.respaldo = "NOUN"

    def entrenar(self, oraciones_anotadas):
        """`oraciones_anotadas`: iterable de listas `[(palabra, etiqueta_upos), …]`."""
        cuentas = defaultdict(Counter)
        globales = Counter()
        for oracion in oraciones_anotadas:
            for palabra, etiqueta in oracion:
                cuentas[palabra.lower()][etiqueta] += 1
                globales[etiqueta] += 1
        self.tabla = {p: c.most_common(1)[0][0] for p, c in cuentas.items()}
        self.respaldo = globales.most_common(1)[0][0]
        return self

    def etiquetar(self, palabra: str) -> str:
        return self.tabla.get(palabra.lower(), self.respaldo)

    def conoce(self, palabra: str) -> bool:
        return palabra.lower() in self.tabla


def profundidad_token(token) -> int:
    """Número de aristas entre el token y la raíz de su árbol de dependencias."""
    pasos, actual = 0, token
    while actual.head != actual and pasos < 100:
        actual = actual.head
        pasos += 1
    return pasos


def profundidad_arbol(doc) -> int:
    """Profundidad máxima del árbol de dependencias del documento."""
    return max((profundidad_token(t) for t in doc), default=0)

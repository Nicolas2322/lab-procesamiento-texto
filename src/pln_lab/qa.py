"""Sistema de preguntas y respuestas extractivo, construido con reglas sobre spaCy.

La tubería tiene tres etapas independientes y evaluables por separado:

1. **Recuperación**: se elige la oración de la noticia más parecida a la pregunta
   (TF-IDF con similitud del coseno o BM25 sobre lemas).
2. **Extracción**: se delimita el tramo de respuesta dentro de esa oración con reglas
   guiadas por el tipo de pregunta (*quién*, *dónde*, *cuánto*, *cuándo*, *qué*), que
   usan las entidades nombradas y la categoría gramatical que asigna spaCy.
3. **Abstención**: si ni la recuperación ni la extracción alcanzan el umbral de
   confianza, el sistema responde con la cadena vacía (las preguntas del conjunto
   marcadas como «sin respuesta» exigen justamente eso).

Las métricas son las de SQuAD (Rajpurkar et al., 2016, 2018): coincidencia exacta y F1
de solapamiento de palabras tras normalizar, tomando el mejor valor entre los alias de
la referencia separados por «|».
"""
from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Normalización y métricas estilo SQuAD
# ---------------------------------------------------------------------------

# Artículos y determinantes que SQuAD descarta en inglés ("a", "an", "the"); aquí se usa
# el equivalente español para que «la absolución» y «absolución» cuenten igual.
ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del"}
_PUNTUACION = re.compile(f"[{re.escape(string.punctuation + '¿¡«»…—–')}]")


def normalizar_respuesta(texto: str) -> str:
    """Minúsculas, sin signos, sin artículos y con los espacios colapsados."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFC", texto).lower()
    texto = _PUNTUACION.sub(" ", texto)
    palabras = [p for p in texto.split() if p not in ARTICULOS]
    return " ".join(palabras)


def coincidencia_exacta(prediccion: str, referencia: str) -> float:
    return float(normalizar_respuesta(prediccion) == normalizar_respuesta(referencia))


def f1_solapamiento(prediccion: str, referencia: str) -> float:
    """F1 sobre la bolsa de palabras normalizadas (métrica oficial de SQuAD)."""
    pred = normalizar_respuesta(prediccion).split()
    ref = normalizar_respuesta(referencia).split()
    if not pred or not ref:
        # Si ambas están vacías la respuesta es correcta (caso «sin respuesta»).
        return float(pred == ref)
    comunes = Counter(pred) & Counter(ref)
    n_comunes = sum(comunes.values())
    if n_comunes == 0:
        return 0.0
    precision, recuperacion = n_comunes / len(pred), n_comunes / len(ref)
    return 2 * precision * recuperacion / (precision + recuperacion)


def separar_alias(referencia) -> list[str]:
    """Las referencias admiten varias formas válidas separadas por «|»."""
    if not isinstance(referencia, str) or not referencia.strip():
        return [""]
    return [a.strip() for a in referencia.split("|") if a.strip()] or [""]


def mejor_puntaje(prediccion: str, referencia, metrica) -> float:
    """Mejor valor de la métrica entre todos los alias de la referencia."""
    return max(metrica(prediccion, alias) for alias in separar_alias(referencia))


# ---------------------------------------------------------------------------
# Recuperación de la oración candidata
# ---------------------------------------------------------------------------

_PATRON_PALABRA = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*", flags=re.UNICODE)


def fichas(texto: str, stopwords: set[str] | None = None) -> list[str]:
    """Tokeniza en minúsculas para la recuperación (números incluidos: llevan información)."""
    tokens = [t.lower() for t in _PATRON_PALABRA.findall(unicodedata.normalize("NFC", texto))]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


class RecuperadorOraciones:
    """Índice de las oraciones de UN documento con dos funciones de puntuación.

    TF-IDF con coseno y BM25 (Robertson y Zaragoza, 2009) se calculan sobre el mismo
    índice para poder compararlos sin cambiar nada más de la tubería.
    """

    def __init__(self, oraciones: list[str], stopwords: set[str] | None = None,
                 k1: float = 1.5, b: float = 0.75):
        self.oraciones = oraciones
        self.stopwords = stopwords or set()
        self.k1, self.b = k1, b
        self.bolsas = [Counter(fichas(o, self.stopwords)) for o in oraciones]
        self.longitudes = [sum(bolsa.values()) for bolsa in self.bolsas]
        self.longitud_media = (sum(self.longitudes) / len(self.longitudes)) if self.longitudes else 0.0
        n = len(oraciones)
        documentos_con = Counter(t for bolsa in self.bolsas for t in bolsa)
        # IDF suavizado para TF-IDF y IDF de Robertson para BM25.
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in documentos_con.items()}
        self.idf_bm25 = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in documentos_con.items()}

    def _vector_tfidf(self, bolsa: Counter) -> dict[str, float]:
        vector = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in bolsa.items()}
        norma = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {t: v / norma for t, v in vector.items()}

    def puntuar(self, pregunta: str, metodo: str = "tfidf") -> list[float]:
        bolsa_pregunta = Counter(fichas(pregunta, self.stopwords))
        if metodo == "tfidf":
            vp = self._vector_tfidf(bolsa_pregunta)
            return [sum(vp.get(t, 0.0) * v for t, v in self._vector_tfidf(bolsa).items())
                    for bolsa in self.bolsas]
        if metodo == "bm25":
            puntajes = []
            for bolsa, longitud in zip(self.bolsas, self.longitudes):
                total = 0.0
                for termino in bolsa_pregunta:
                    frecuencia = bolsa.get(termino, 0)
                    if not frecuencia:
                        continue
                    denominador = frecuencia + self.k1 * (
                        1 - self.b + self.b * longitud / (self.longitud_media or 1.0))
                    total += self.idf_bm25.get(termino, 0.0) * frecuencia * (self.k1 + 1) / denominador
                puntajes.append(total)
            # Se normaliza a [0, 1] para que el umbral de abstención sea comparable con TF-IDF.
            maximo = max(puntajes) if puntajes else 0.0
            return [p / maximo if maximo else 0.0 for p in puntajes]
        raise ValueError(f"Método de recuperación desconocido: {metodo}")

    def mejor(self, pregunta: str, metodo: str = "tfidf") -> tuple[int, float]:
        puntajes = self.puntuar(pregunta, metodo)
        if not puntajes:
            return -1, 0.0
        indice = max(range(len(puntajes)), key=puntajes.__getitem__)
        return indice, puntajes[indice]


# ---------------------------------------------------------------------------
# Clasificación del tipo de pregunta
# ---------------------------------------------------------------------------

_DISPARADORES = (
    ("persona", (r"\bqui[eé]n(?:es)?\b",)),
    ("cantidad", (r"\bcu[aá]nt[oa]s?\b", r"\ba cu[aá]nto\b", r"\bqu[eé] precio\b",
                  r"\bqu[eé] porcentaje\b", r"\bcu[aá]l es el n[uú]mero\b")),
    ("fecha", (r"\bcu[aá]ndo\b", r"\bhasta cu[aá]ndo\b", r"\bqu[eé] (?:d[ií]a|fecha|a[nñ]o|mes)\b",
               r"\ben qu[eé] (?:d[ií]a|fecha|a[nñ]o|mes)\b")),
    ("lugar", (r"\bd[oó]nde\b", r"\bqu[eé] (?:pa[ií]s|ciudad|lugar|aeropuerto|regi[oó]n|provincia)\b",
               r"\bde qu[eé] (?:pa[ií]s|ciudad|lugar|aeropuerto)\b")),
    ("causa", (r"\bpor qu[eé]\b", r"\bpor qu[eé] raz[oó]n\b", r"\bqu[eé] causa\b", r"\ba qu[eé] se debe\b")),
    ("otro", (r"\bqu[eé]\b", r"\bcu[aá]l(?:es)?\b", r"\bc[oó]mo\b")),
)


def clasificar_pregunta(pregunta: str) -> str:
    """Devuelve el tipo esperado de respuesta a partir del pronombre interrogativo."""
    texto = " " + pregunta.lower() + " "
    for tipo, patrones in _DISPARADORES:
        if any(re.search(p, texto) for p in patrones):
            return tipo
    return "otro"


# ---------------------------------------------------------------------------
# Extracción del tramo de respuesta
# ---------------------------------------------------------------------------

NUMEROS_PALABRA = {
    "un", "una", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
    "once", "doce", "trece", "catorce", "quince", "dieciséis", "veinte", "treinta", "cuarenta",
    "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien", "ciento", "mil", "medio",
    "media", "docena", "centenar", "millar",
}
ESCALAS = {"millones", "millón", "millon", "mil", "miles", "billones", "billón", "centenar",
           "centenares", "docenas", "decenas", "millares"}
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "setiembre", "octubre", "noviembre", "diciembre")
_MES = "|".join(MESES)
_RE_FECHA = re.compile(
    rf"(?:el\s+)?(?:pr[oó]ximo\s+|pasado\s+)?\b\d{{1,2}}\s+de\s+(?:{_MES})(?:\s+de\s+(?:19|20)?\d{{2}})?"
    rf"|(?:el\s+)?mes\s+de\s+(?:{_MES})(?:\s+de\s+(?:19|20)\d{{2}})?"
    rf"|\b(?:{_MES})\s+de\s+(?:19|20)\d{{2}}"
    rf"|\b(?:{_MES})\s+(?:pr[oó]ximo|pasado)\b"
    rf"|\b(?:19|20)\d{{2}}\b"
    rf"|\b(?:{_MES})\b",
    flags=re.IGNORECASE)
# Nexos causales: el tramo de respuesta empieza justo después del nexo.
_NEXOS_CAUSA = ("porque", "debido a", "a causa de", "por culpa de", "gracias a", "ya que",
                "puesto que", "como consecuencia de", "a raíz de", "tras")
_VERBOS_CAUSA = ("obligó", "obliga", "provocó", "provoca", "causó", "causa", "impidió", "impide",
                 "ocasionó", "ocasiona", "generó")

_ETIQUETAS_LUGAR = {"LOC", "GPE"}
_ETIQUETAS_PERSONA = {"PER", "PERSON"}


def _contenido(texto: str, stopwords: set[str]) -> set[str]:
    return {t for t in fichas(texto, stopwords) if len(t) > 1}


def _candidatos_entidad(doc, etiquetas) -> list:
    return [e for e in doc.ents if e.label_ in etiquetas]


def _candidatos_cantidad(doc) -> list:
    """Tramos que empiezan en un numeral y se extienden sobre la unidad de medida.

    La extensión cruza una preposición «de»/«por» solo si la palabra anterior es una escala
    (*millones*, *miles*): así se captura «600 millones de dólares» sin arrastrar el
    complemento entero de «35 piezas de diferentes formatos».
    """
    candidatos = []
    i = 0
    while i < len(doc):
        token = doc[i]
        es_numeral = token.like_num or token.text.lower() in NUMEROS_PALABRA
        if not es_numeral or token.pos_ in {"PRON", "DET"} and token.text.lower() in {"un", "una"} and i + 1 < len(doc) and doc[i + 1].pos_ == "NOUN" and not doc[i + 1].text.lower() in ESCALAS:
            # «una granada» no es una cantidad: se exige que el numeral sea NUM o una escala.
            if not (token.pos_ == "NUM" or token.like_num or token.text.lower() in ESCALAS):
                i += 1
                continue
        if not es_numeral:
            i += 1
            continue
        inicio, fin = i, i + 1
        # Modificadores previos habituales: «al menos 28 muertos», «unos 1.000 kilómetros».
        if inicio > 0 and doc[inicio - 1].text.lower() in {"unos", "unas", "casi", "más", "menos", "cerca"}:
            inicio -= 1
        while fin < len(doc):
            siguiente = doc[fin]
            texto_sig = siguiente.text.lower()
            anterior = doc[fin - 1].text.lower()
            if texto_sig in ESCALAS or siguiente.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"} and siguiente.pos_ != "PUNCT":
                if siguiente.pos_ == "ADJ" and fin - inicio > 3:
                    break
                fin += 1
                continue
            if texto_sig in {"de", "por"} and anterior in ESCALAS and fin + 1 < len(doc):
                fin += 1
                continue
            if texto_sig == "%" or texto_sig in {"por"} and doc[fin:fin + 2].text.lower() == "por ciento":
                fin += 1
                continue
            break
        candidatos.append(doc[inicio:fin])
        i = max(fin, i + 1)
    return candidatos


def _candidatos_fecha(doc) -> list:
    """Tramos de fecha localizados con expresiones regulares y realineados a tokens."""
    candidatos = []
    for coincidencia in _RE_FECHA.finditer(doc.text):
        tramo = doc.char_span(coincidencia.start(), coincidencia.end(), alignment_mode="expand")
        if tramo is not None and tramo.text.strip():
            candidatos.append(tramo)
    return candidatos


def _candidatos_nominal(doc) -> list:
    """Sintagmas nominales. Usa `noun_chunks` si el modelo los ofrece; si no, los reconstruye
    agrupando determinante + sustantivo + modificadores a partir de la categoría gramatical."""
    def util(tramo) -> bool:
        # Un sintagma sin sustantivo ni nombre propio («que», «los cuales») no es respuesta.
        return any(t.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"} for t in tramo)

    try:
        trozos = [t for t in doc.noun_chunks if util(t)]
        if trozos:
            return trozos
    except (NotImplementedError, ValueError):
        pass
    candidatos, i = [], 0
    while i < len(doc):
        if doc[i].pos_ in {"NOUN", "PROPN"} or (doc[i].pos_ == "DET" and i + 1 < len(doc)
                                                and doc[i + 1].pos_ in {"NOUN", "PROPN", "ADJ"}):
            inicio = i
            if doc[i].pos_ == "DET":
                i += 1
            while i < len(doc) and doc[i].pos_ in {"NOUN", "PROPN", "ADJ"}:
                i += 1
            if i > inicio:
                candidatos.append(doc[inicio:i])
                continue
        i += 1
    return candidatos


def _candidatos_causa(doc) -> list:
    """Tramo que sigue a un nexo causal, o sujeto del verbo causativo («la falta de
    combustible obligó a…»)."""
    texto = doc.text
    candidatos = []
    minuscula = texto.lower()
    for nexo in _NEXOS_CAUSA:
        for coincidencia in re.finditer(rf"\b{re.escape(nexo)}\b", minuscula):
            inicio = coincidencia.end()
            fin = len(texto)
            for corte in (",", ";", ".", " pero ", " aunque ", " y "):
                posicion = minuscula.find(corte, inicio + 1)
                if posicion != -1:
                    fin = min(fin, posicion)
            tramo = doc.char_span(inicio, fin, alignment_mode="contract")
            if tramo is not None and len(tramo) >= 2:
                candidatos.append(_recortar_nexo(tramo))
    for verbo in _VERBOS_CAUSA:
        posicion = minuscula.find(f" {verbo} ")
        if posicion == -1:
            continue
        # Sujeto: lo que hay entre el corte anterior y el verbo.
        inicio = max(minuscula.rfind(sep, 0, posicion) + 1 for sep in (",", ";", ".", ":"))
        tramo = doc.char_span(inicio, posicion, alignment_mode="contract")
        if tramo is not None and 1 <= len(tramo) <= 8:
            candidatos.append(_recortar_nexo(tramo))
    # Una causa se expresa con un sintagma nominal: los tramos sin sustantivo («no», «así»)
    # son ruido del recorte por signos de puntuación.
    return [t for t in candidatos if any(x.pos_ in {"NOUN", "PROPN"} for x in t)]


_ARRANQUES_SOBRANTES = {"pero", "y", "aunque", "sin", "embargo", "que", "sino", "mientras", "cuando"}


def _recortar_nexo(tramo):
    """Quita conjunciones iniciales del tramo causal: «pero la falta de combustible» →
    «la falta de combustible»."""
    inicio = tramo.start
    while inicio < tramo.end and tramo.doc[inicio].text.lower() in _ARRANQUES_SOBRANTES:
        inicio += 1
    return tramo.doc[inicio:tramo.end] if inicio < tramo.end else tramo


def _elegir(candidatos, palabras_pregunta: set[str], stopwords: set[str], doc):
    """De entre los candidatos, descarta los que repiten literalmente la pregunta y
    prefiere el que tiene más palabras de la pregunta cerca (ponderadas por la distancia)."""
    utiles = []
    for tramo in candidatos:
        propias = _contenido(tramo.text, stopwords)
        if propias and propias <= palabras_pregunta:
            continue  # el candidato ya aparece en la pregunta: no puede ser la respuesta
        utiles.append(tramo)
    if not utiles:
        utiles = list(candidatos)
    if not utiles:
        return None, 0.0
    posiciones = {}
    for token in doc:
        clave = token.text.lower()
        if clave in palabras_pregunta:
            posiciones.setdefault(clave, []).append(token.i)

    def cercania(tramo):
        centro = (tramo.start + tramo.end) / 2
        return sum(max(1.0 / (1 + abs(centro - i)) for i in indices)
                   for indices in posiciones.values())

    mejor = max(utiles, key=cercania)
    # Confianza de la extracción: cuántas palabras de la pregunta hay alrededor.
    confianza = min(1.0, cercania(mejor) / max(1, len(posiciones)) * 2)
    return mejor, confianza


@dataclass
class Respuesta:
    """Salida del sistema para una pregunta."""

    texto: str = ""
    tipo: str = "otro"
    indice_oracion: int = -1
    oracion: str = ""
    puntaje_recuperacion: float = 0.0
    confianza_extraccion: float = 0.0
    cobertura: float = 0.0
    confianza: float = 0.0
    abstenido: bool = False
    motivo: str = ""


@dataclass
class SistemaQA:
    """Tubería completa: recuperación + extracción por reglas + abstención.

    `nlp` debe ser un modelo de spaCy con etiquetador, analizador sintáctico y NER
    (`es_core_news_md`). `umbral` es la confianza mínima para no abstenerse.
    """

    nlp: object
    stopwords: set = field(default_factory=set)
    metodo_recuperacion: str = "tfidf"
    umbral: float = 0.40
    k_oraciones: int = 3          # oraciones candidatas que se reordenan con la extracción
    pesos: tuple = (0.35, 0.35, 0.30)   # (recuperación, extracción, cobertura léxica)
    _cache_docs: dict = field(default_factory=dict, repr=False)

    def _doc(self, texto: str):
        if texto not in self._cache_docs:
            self._cache_docs[texto] = self.nlp(texto)
        return self._cache_docs[texto]

    def extraer(self, oracion: str, pregunta: str, tipo: str):
        """Aplica la regla correspondiente al tipo de pregunta sobre la oración recuperada."""
        doc = self._doc(oracion)
        palabras_pregunta = _contenido(pregunta, self.stopwords)
        if tipo == "persona":
            candidatos = _candidatos_entidad(doc, _ETIQUETAS_PERSONA) or _candidatos_nominal(doc)
        elif tipo == "lugar":
            candidatos = (_candidatos_entidad(doc, _ETIQUETAS_LUGAR)
                          or _candidatos_entidad(doc, {"ORG", "MISC"}) or _candidatos_nominal(doc))
        elif tipo == "cantidad":
            candidatos = _candidatos_cantidad(doc)
        elif tipo == "fecha":
            candidatos = _candidatos_fecha(doc)
        elif tipo == "causa":
            candidatos = _candidatos_causa(doc)
        else:
            candidatos = _candidatos_nominal(doc)
        if not candidatos:
            return None, 0.0
        return _elegir(candidatos, palabras_pregunta, self.stopwords, doc)

    def responder(self, oraciones: list[str], pregunta: str,
                  recuperador: RecuperadorOraciones | None = None) -> Respuesta:
        """Reordena las `k_oraciones` mejores por recuperación usando la extracción.

        La oración más parecida a la pregunta no siempre contiene la respuesta (una pregunta
        de causa comparte vocabulario con la oración que enuncia el hecho, no con la que lo
        explica). Por eso se extrae sobre varias oraciones y se elige la combinación
        oración + tramo con mayor confianza.
        """
        tipo = clasificar_pregunta(pregunta)
        recuperador = recuperador or RecuperadorOraciones(oraciones, self.stopwords)
        puntajes = recuperador.puntuar(pregunta, self.metodo_recuperacion)
        if not puntajes:
            return Respuesta(tipo=tipo, abstenido=True, motivo="documento vacío")
        orden = sorted(range(len(puntajes)), key=puntajes.__getitem__, reverse=True)[:self.k_oraciones]
        contenido_pregunta = _contenido(pregunta, self.stopwords)
        peso_r, peso_e, peso_c = self.pesos

        mejor = None
        for indice in orden:
            oracion = oraciones[indice]
            tramo, confianza_extraccion = self.extraer(oracion, pregunta, tipo)
            # Cobertura léxica: qué fracción de las palabras de contenido de la pregunta
            # aparece en la oración. Es la señal que más discrimina las preguntas sin respuesta.
            propias = _contenido(oracion, self.stopwords)
            cobertura = (len(contenido_pregunta & propias) / len(contenido_pregunta)
                         if contenido_pregunta else 0.0)
            confianza = peso_r * puntajes[indice] + peso_e * confianza_extraccion + peso_c * cobertura
            candidata = Respuesta(
                texto="" if tramo is None else tramo.text.strip(" ,.;:«»\"'()"),
                tipo=tipo, indice_oracion=indice, oracion=oracion,
                puntaje_recuperacion=puntajes[indice], confianza_extraccion=confianza_extraccion,
                cobertura=cobertura, confianza=0.0 if tramo is None else confianza,
                motivo="" if tramo is not None else f"sin candidato de tipo {tipo}")
            if mejor is None or candidata.confianza > mejor.confianza:
                mejor = candidata
        if not mejor.texto:
            mejor.abstenido, mejor.texto = True, ""
            return mejor
        if mejor.confianza < self.umbral:
            mejor.abstenido, mejor.texto = True, ""
            mejor.motivo = "confianza por debajo del umbral"
        return mejor


def evaluar_qa(predicciones: list[str], referencias: list[str]) -> dict:
    """Coincidencia exacta y F1 promedios (convención de SQuAD, en porcentaje)."""
    if not predicciones:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    em = [mejor_puntaje(p, r, coincidencia_exacta) for p, r in zip(predicciones, referencias)]
    f1 = [mejor_puntaje(p, r, f1_solapamiento) for p, r in zip(predicciones, referencias)]
    return {"em": 100 * sum(em) / len(em), "f1": 100 * sum(f1) / len(f1), "n": len(em)}

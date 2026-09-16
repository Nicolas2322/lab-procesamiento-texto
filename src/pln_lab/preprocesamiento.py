"""Normalización, stopwords, stemming y lematización para texto en español.

Cada paso es una función pura (texto -> texto o tokens -> tokens) para que los notebooks
puedan aplicar los pasos de uno en uno y medir el efecto de cada decisión por separado.
La clase ``Normalizador`` los encadena con una configuración explícita.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tokenización
# ---------------------------------------------------------------------------

# Orden de las alternativas: marcadores (<NUM>, <URL>...), palabras con guion o apóstrofo,
# números con separadores decimales o de miles y, por último, cualquier signo aislado.
_PATRON_TOKEN = re.compile(
    r"<[A-Z]+>"
    r"|[^\W\d_]+(?:[-'][^\W\d_]+)*"
    r"|\d+(?:[.,:]\d+)*"
    r"|[^\w\s]",
    flags=re.UNICODE,
)
_PATRON_PALABRA = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$", flags=re.UNICODE)


def tokenizar_espacios(texto: str) -> list[str]:
    """Tokenización ingenua: separa solo por espacios (los signos quedan pegados)."""
    return texto.split()


def tokenizar(texto: str) -> list[str]:
    """Tokenización por expresiones regulares que separa los signos de puntuación."""
    return _PATRON_TOKEN.findall(texto)


def es_palabra(token: str) -> bool:
    return bool(_PATRON_PALABRA.match(token))


# ---------------------------------------------------------------------------
# Normalización a nivel de carácter y de token
# ---------------------------------------------------------------------------

def normalizar_unicode(texto: str, forma: str = "NFC") -> str:
    """Unifica representaciones equivalentes ('é' precompuesta vs. 'e' + tilde combinante)."""
    return unicodedata.normalize(forma, texto)


def quitar_tildes(texto: str, conservar_enie: bool = True) -> str:
    """Elimina diacríticos. Por defecto conserva la ñ (y la Ñ), que en español es una letra."""
    resultado = []
    for caracter in unicodedata.normalize("NFD", texto):
        if unicodedata.category(caracter) != "Mn":
            resultado.append(caracter)
            continue
        # Virgulilla (U+0303) sobre n/N: se conserva para no convertir 'año' en 'ano'.
        if conservar_enie and caracter == "\u0303" and resultado and resultado[-1] in "nN":
            resultado.append(caracter)
    return unicodedata.normalize("NFC", "".join(resultado))


_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_MENCION = re.compile(r"(?<!\w)@\w+")
_HASHTAG = re.compile(r"(?<!\w)#(\w+)")
_FECHA = re.compile(r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?!\d)")
_NUMERO = re.compile(r"(?<![\w<])\d+(?:[.,:]\d+)*(?![\w>])")
# Consonantes que en español aparecen dobles legítimamente: acción, llamar, carro.
_DOBLES_VALIDAS = "clrCLR"
_ALARGAMIENTO = re.compile(r"([^\W\d_])\1{2,}", flags=re.UNICODE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]+"
)


def reducir_alargamientos(texto: str) -> str:
    """'holaaaa' -> 'hola', 'carrrro' -> 'carro'. Heurística: 3 o más repeticiones se reducen a 2
    si la consonante admite duplicación en español (c, l, r) y a 1 en los demás casos.
    Limitación conocida: 'leeeer' -> 'ler' (las vocales dobles legítimas se pierden)."""

    def _reemplazo(m: re.Match) -> str:
        letra = m.group(1)
        return letra * 2 if letra in _DOBLES_VALIDAS else letra

    return _ALARGAMIENTO.sub(_reemplazo, texto)


def reemplazar_entidades_ruido(texto: str, numeros: bool = True) -> str:
    """Sustituye URLs, correos, menciones, emojis, fechas y números por marcadores genéricos."""
    texto = _URL.sub(" <URL> ", texto)
    texto = _EMAIL.sub(" <EMAIL> ", texto)
    texto = _MENCION.sub(" <USUARIO> ", texto)
    # Los hashtags conservan la palabra; si vienen en CamelCase se separan: #VacunaciónYa -> Vacunación Ya
    texto = _HASHTAG.sub(lambda m: " " + re.sub(r"(?<=[a-záéíóúüñ])(?=[A-ZÁÉÍÓÚÜÑ])", " ", m.group(1)) + " ", texto)
    texto = _EMOJI.sub(" <EMOJI> ", texto)
    if numeros:
        texto = _FECHA.sub(" <FECHA> ", texto)
        texto = _NUMERO.sub(" <NUM> ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _minusculas_protegiendo_marcadores(texto: str) -> str:
    partes = re.split(r"(<[A-Z]+>)", texto)
    return "".join(p if re.fullmatch(r"<[A-Z]+>", p) else p.lower() for p in partes)


_CABECERA_AGENCIA = re.compile(r"^.{0,120}?\((?:EFE\w*|AFP|AP|Reuters)\)\s*\.?\s*-?\s*")


def quitar_cabecera_agencia(texto: str) -> str:
    """Elimina la cabecera de las notas de agencia: 'Madrid, 25 may (EFE). - '.
    Es ruido de dominio: se repite en todas las noticias y no aporta contenido."""
    return _CABECERA_AGENCIA.sub("", texto, count=1)


@dataclass
class Normalizador:
    """Cadena de normalización configurable.

    Parámetros por defecto pensados para noticias en español: minúsculas, números
    reemplazados, signos eliminados y **tildes conservadas** (quitar tildes fusiona
    palabras distintas como 'más'/'mas' o 'él'/'el'; ver notebook 01, sección 3).
    """

    quitar_cabecera: bool = True
    minusculas: bool = True
    quitar_diacriticos: bool = False
    reemplazar_numeros: bool = True
    reducir_repeticiones: bool = True
    solo_palabras: bool = True

    def __call__(self, texto: str) -> list[str]:
        texto = normalizar_unicode(texto)
        if self.quitar_cabecera:
            texto = quitar_cabecera_agencia(texto)
        if self.reducir_repeticiones:
            texto = reducir_alargamientos(texto)
        texto = reemplazar_entidades_ruido(texto, numeros=self.reemplazar_numeros)
        if self.minusculas:
            # Los marcadores <NUM>, <URL>... se protegen de las minúsculas.
            texto = _minusculas_protegiendo_marcadores(texto)
        if self.quitar_diacriticos:
            texto = quitar_tildes(texto)
        tokens = tokenizar(texto)
        if self.solo_palabras:
            tokens = [t for t in tokens if es_palabra(t) or t.startswith("<")]
        return tokens


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

NEGACIONES = {"no", "ni", "nunca", "jamás", "tampoco", "nada", "nadie", "ninguno", "ninguna", "sin"}


def stopwords_nltk() -> set[str]:
    import nltk

    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    return set(stopwords.words("spanish"))


def stopwords_spacy() -> set[str]:
    from spacy.lang.es.stop_words import STOP_WORDS

    return set(STOP_WORDS)


def quitar_stopwords(tokens: list[str], lista: set[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in lista]


# ---------------------------------------------------------------------------
# Stemming y lematización
# ---------------------------------------------------------------------------

class Stemmer:
    """Envoltorio del Snowball español con caché (el vocabulario es mucho menor que el corpus)."""

    def __init__(self, idioma: str = "spanish"):
        from nltk.stem.snowball import SnowballStemmer

        self._stemmer = SnowballStemmer(idioma)
        self._cache: dict[str, str] = {}

    def __call__(self, palabra: str) -> str:
        if palabra not in self._cache:
            self._cache[palabra] = palabra if palabra.startswith("<") else self._stemmer.stem(palabra)
        return self._cache[palabra]

    def aplicar(self, tokens: list[str]) -> list[str]:
        return [self(t) for t in tokens]


def cargar_spacy(modelo: str = "es_core_news_md", deshabilitar=("ner", "parser")):
    import spacy

    try:
        return spacy.load(modelo, disable=list(deshabilitar))
    except OSError as error:  # modelo no instalado
        raise OSError(
            f"Falta el modelo {modelo}. Instálelo con: python -m spacy download {modelo}"
        ) from error


def lematizar_textos(nlp, textos, batch_size: int = 32):
    """Lematiza una colección de textos con spaCy. Devuelve, por texto, una lista de
    tuplas (token, lema, categoría gramatical universal)."""
    salida = []
    for doc in nlp.pipe(textos, batch_size=batch_size):
        salida.append([(t.text, t.lemma_, t.pos_) for t in doc])
    return salida


# ---------------------------------------------------------------------------
# Métricas de vocabulario
# ---------------------------------------------------------------------------

def resumen_vocabulario(tokens: list[str]) -> dict:
    conteo = Counter(tokens)
    n_tokens, n_tipos = sum(conteo.values()), len(conteo)
    return {
        "tokens": n_tokens,
        "tipos": n_tipos,
        "ttr": n_tipos / n_tokens if n_tokens else 0.0,
        "hapax": sum(1 for c in conteo.values() if c == 1),
    }


def colisiones(vocabulario, transformacion) -> dict[str, set[str]]:
    """Agrupa los tipos que una transformación convierte en la misma forma.
    Devuelve solo los grupos con más de un miembro (las 'fusiones')."""
    grupos: dict[str, set[str]] = {}
    for palabra in vocabulario:
        grupos.setdefault(transformacion(palabra), set()).add(palabra)
    return {forma: miembros for forma, miembros in grupos.items() if len(miembros) > 1}

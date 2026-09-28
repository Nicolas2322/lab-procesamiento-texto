"""Generación de texto: tokenizador compatible con Keras, modelo de n-gramas y
métricas de diversidad.

El laboratorio de clase (*Course 3 – Week 4* de Moroney) construye el modelo de
predicción de la siguiente palabra con `keras.preprocessing.text.Tokenizer`. Keras 3
—la versión que acompaña a TensorFlow 2.21— eliminó ese módulo, de modo que aquí se
reimplementa la parte de su interfaz que el laboratorio usa (`fit_on_textos`,
`textos_a_secuencias`, `word_index`, `index_word`, `num_words`, `oov_token`). Así el
notebook sigue el mismo procedimiento sin depender de una API retirada.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_PATRON_PALABRA = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)

INICIO, FIN = "<inicio>", "<fin>"


# ---------------------------------------------------------------------------
# Tokenizador (sustituto de keras.preprocessing.text.Tokenizer)
# ---------------------------------------------------------------------------

class TokenizadorPalabras:
    """Asigna un índice entero a cada palabra, ordenada por frecuencia descendente.

    Réplica del comportamiento de `Tokenizer` de Keras: el índice 0 queda reservado
    para el relleno (`pad`), el token fuera de vocabulario recibe el índice 1 y
    `num_words` recorta el vocabulario a las palabras más frecuentes.
    """

    def __init__(self, num_words: int | None = None, oov_token: str = "<oov>", minusculas: bool = True):
        self.num_words = num_words
        self.oov_token = oov_token
        self.minusculas = minusculas
        self.conteos: Counter = Counter()
        self.word_index: dict[str, int] = {}
        self.index_word: dict[int, str] = {}

    def tokenizar(self, texto: str) -> list[str]:
        if self.minusculas:
            texto = texto.lower()
        return _PATRON_PALABRA.findall(texto)

    def fit_on_textos(self, textos) -> "TokenizadorPalabras":
        for texto in textos:
            self.conteos.update(self.tokenizar(texto))
        # El token OOV ocupa el índice 1 para que el 0 quede libre como relleno.
        ordenadas = [p for p, _ in self.conteos.most_common()]
        self.word_index = {self.oov_token: 1}
        for i, palabra in enumerate(ordenadas, start=2):
            self.word_index[palabra] = i
        self.index_word = {i: p for p, i in self.word_index.items()}
        return self

    @property
    def tamano_vocabulario(self) -> int:
        """Tamaño de la capa de salida: índices 0…máximo índice utilizable."""
        total = len(self.word_index) + 1
        return total if self.num_words is None else min(total, self.num_words + 1)

    def indice(self, palabra: str) -> int:
        i = self.word_index.get(palabra.lower() if self.minusculas else palabra, 1)
        return i if (self.num_words is None or i <= self.num_words) else 1

    def textos_a_secuencias(self, textos) -> list[list[int]]:
        return [[self.indice(p) for p in self.tokenizar(t)] for t in textos]

    def secuencia_a_texto(self, secuencia) -> str:
        return " ".join(self.index_word.get(int(i), self.oov_token) for i in secuencia if int(i) != 0)


# ---------------------------------------------------------------------------
# Modelo de n-gramas con suavizado
# ---------------------------------------------------------------------------

class ModeloNGramas:
    """Modelo de lenguaje de n-gramas con suavizado add-k (Lidstone).

    Se implementa a mano para dejar explícito el cálculo que después se compara con
    la red neuronal: la probabilidad de una palabra depende solo de las n−1
    anteriores, y el suavizado reparte masa hacia los n-gramas no observados para que
    la perplejidad sea finita.
    """

    def __init__(self, n: int = 2, k: float = 0.1):
        if n < 2:
            raise ValueError("n debe ser 2 o mayor")
        self.n, self.k = n, k
        self.conteos: dict[tuple, Counter] = defaultdict(Counter)
        self.total_contexto: dict[tuple, int] = defaultdict(int)
        self.vocabulario: set[str] = set()

    def _con_fronteras(self, tokens: list[str]) -> list[str]:
        return [INICIO] * (self.n - 1) + list(tokens) + [FIN]

    def entrenar(self, documentos_tokenizados) -> "ModeloNGramas":
        for tokens in documentos_tokenizados:
            secuencia = self._con_fronteras(tokens)
            self.vocabulario.update(tokens)
            for i in range(self.n - 1, len(secuencia)):
                contexto = tuple(secuencia[i - self.n + 1:i])
                self.conteos[contexto][secuencia[i]] += 1
                self.total_contexto[contexto] += 1
        self.vocabulario.add(FIN)
        self._lista_vocabulario = sorted(self.vocabulario)
        return self

    @property
    def tamano_vocabulario(self) -> int:
        return len(self.vocabulario)

    def probabilidad(self, contexto: tuple, palabra: str) -> float:
        v = self.tamano_vocabulario
        numerador = self.conteos[contexto][palabra] + self.k
        return numerador / (self.total_contexto[contexto] + self.k * v)

    def distribucion(self, contexto: tuple, prohibidos=()) -> tuple[list[str], np.ndarray]:
        """Distribución sobre el vocabulario para un contexto dado.

        Para no recorrer todo el vocabulario en cada paso, se devuelven solo las
        continuaciones observadas más una masa agregada; el muestreo usa esta versión
        restringida (equivale a muestrear de la distribución suavizada salvo por la
        cola de n-gramas nunca vistos, que con add-k = 0,1 es despreciable).
        """
        observadas = self.conteos[contexto]
        palabras = [p for p in observadas if p not in prohibidos]
        if not palabras:
            return [], np.array([])
        probs = np.array([self.probabilidad(contexto, p) for p in palabras])
        return palabras, probs / probs.sum()

    def log_probabilidad(self, tokens: list[str]) -> tuple[float, int]:
        """Log-probabilidad en base 2 de una secuencia y número de predicciones."""
        secuencia = self._con_fronteras(tokens)
        total, n_pred = 0.0, 0
        for i in range(self.n - 1, len(secuencia)):
            contexto = tuple(secuencia[i - self.n + 1:i])
            total += math.log2(self.probabilidad(contexto, secuencia[i]))
            n_pred += 1
        return total, n_pred

    def perplejidad(self, documentos_tokenizados) -> float:
        suma, n = 0.0, 0
        for tokens in documentos_tokenizados:
            lp, k = self.log_probabilidad(tokens)
            suma += lp
            n += k
        return float(2 ** (-suma / n)) if n else float("nan")

    def generar(self, semilla: str, n_palabras: int = 20, temperatura: float = 1.0,
                top_k: int | None = None, greedy: bool = False, rng=None,
                prohibidos=("<oov>",)) -> str:
        """Genera texto a partir de una semilla.

        `prohibidos` excluye del muestreo el token fuera de vocabulario: aparece en el
        modelo (y cuenta para la perplejidad) pero no es una palabra del español, así
        que mostrarlo en las generaciones solo añadiría ruido a la comparación.
        """
        rng = rng or np.random.default_rng(0)
        tokens = _PATRON_PALABRA.findall(semilla.lower())
        historial = [INICIO] * (self.n - 1) + tokens
        generadas: list[str] = []
        for _ in range(n_palabras):
            contexto = tuple(historial[-(self.n - 1):])
            palabras, probs = self.distribucion(contexto, prohibidos)
            if len(palabras) == 0:  # contexto no visto: se reinicia al contexto inicial
                contexto = tuple([INICIO] * (self.n - 1))
                palabras, probs = self.distribucion(contexto, prohibidos)
                if len(palabras) == 0:
                    break
            siguiente = elegir(palabras, probs, temperatura, top_k, greedy, rng)
            if siguiente == FIN:
                break
            generadas.append(siguiente)
            historial.append(siguiente)
        return " ".join(tokens + generadas)


def elegir(candidatos, probs: np.ndarray, temperatura: float = 1.0, top_k: int | None = None,
           greedy: bool = False, rng=None):
    """Selecciona un elemento aplicando temperatura y, opcionalmente, recorte top-k."""
    rng = rng or np.random.default_rng(0)
    probs = np.asarray(probs, dtype=np.float64)
    if greedy:
        return candidatos[int(np.argmax(probs))]
    logits = np.log(np.clip(probs, 1e-12, None)) / max(temperatura, 1e-6)
    ajustadas = np.exp(logits - logits.max())
    ajustadas /= ajustadas.sum()
    if top_k is not None and top_k < len(ajustadas):
        corte = np.argsort(ajustadas)[-top_k:]
        mascara = np.zeros_like(ajustadas)
        mascara[corte] = ajustadas[corte]
        ajustadas = mascara / mascara.sum()
    return candidatos[int(rng.choice(len(ajustadas), p=ajustadas))]


# ---------------------------------------------------------------------------
# Generación con el modelo neuronal
# ---------------------------------------------------------------------------

def generar_con_red(modelo, tokenizador: TokenizadorPalabras, semilla: str, n_palabras: int = 20,
                    maxlen: int = 12, temperatura: float = 1.0, top_k: int | None = None,
                    greedy: bool = False, rng=None, indices_prohibidos=(0, 1)) -> str:
    """Predice palabra a palabra con el modelo de Keras, igual que el laboratorio de clase.

    El bucle es el de Moroney (*Course 3 – Week 4*): se tokeniza la semilla, se rellena a
    `maxlen`, se predice la distribución de la siguiente palabra y se realimenta. La
    diferencia es que aquí la elección no es siempre `argmax`: se puede aplicar
    temperatura y recorte top-k para estudiar su efecto.
    """
    from keras.utils import pad_sequences

    rng = rng or np.random.default_rng(0)
    tokens = [tokenizador.indice(p) for p in tokenizador.tokenizar(semilla)]
    palabras = list(tokenizador.tokenizar(semilla))
    for _ in range(n_palabras):
        entrada = pad_sequences([tokens[-maxlen:]], maxlen=maxlen)
        probs = modelo.predict(entrada, verbose=0)[0].astype(np.float64)
        probs[list(indices_prohibidos)] = 0.0  # relleno y token fuera de vocabulario
        probs /= probs.sum()
        candidatos = np.arange(len(probs))
        elegido = int(elegir(candidatos, probs, temperatura, top_k, greedy, rng))
        tokens.append(elegido)
        palabras.append(tokenizador.index_word.get(elegido, tokenizador.oov_token))
    return " ".join(palabras)


# ---------------------------------------------------------------------------
# Métricas de diversidad y repetición
# ---------------------------------------------------------------------------

def distinct_n(textos, n: int = 1) -> float:
    """Proporción de n-gramas distintos sobre el total de n-gramas generados."""
    total, distintos = 0, set()
    for texto in textos:
        tokens = _PATRON_PALABRA.findall(texto.lower())
        gramas = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(gramas)
        distintos.update(gramas)
    return len(distintos) / total if total else 0.0


def tasa_repeticion(textos, n: int = 3) -> float:
    """Fracción de n-gramas que se repiten dentro del mismo texto (degeneración)."""
    repetidos, total = 0, 0
    for texto in textos:
        tokens = _PATRON_PALABRA.findall(texto.lower())
        gramas = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        conteo = Counter(gramas)
        total += len(gramas)
        repetidos += sum(c - 1 for c in conteo.values() if c > 1)
    return repetidos / total if total else 0.0


def longitud_media(textos) -> float:
    return float(np.mean([len(_PATRON_PALABRA.findall(t)) for t in textos])) if textos else 0.0

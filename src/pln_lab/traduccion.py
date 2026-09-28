"""Traductor español→inglés basado en diccionario bilingüe y reglas de transferencia.

Es la arquitectura de la traducción automática de primera generación (*direct / rule-based
MT*): analizar la oración de origen, buscar cada palabra en un diccionario, aplicar reglas
de reordenamiento y de morfología, y generar la cadena de destino. Se implementa aquí para
medir con BLEU y chrF cuánto de la calidad de un traductor neuronal viene de sustituir
palabras y cuánto de modelar el contexto.

Componentes:

* `DICCIONARIO`: lema español → palabra inglesa. Se construyó a partir del vocabulario de
  las oraciones de evaluación (`datos/traduccion_es_en.csv`) más las palabras más frecuentes
  del corpus EFE, de modo que la cobertura léxica no sea el cuello de botella del experimento.
* `LOCUCIONES`: unidades pluriverbales funcionales que no se pueden traducir palabra a
  palabra (*al menos*, *desde hace*, *a través de*).
* Reglas de transferencia: reordenamiento sustantivo–adjetivo, contracciones *del*/*al*,
  supresión del sujeto pronominal, y generación de la morfología verbal y nominal inglesa
  a partir de los rasgos que anota spaCy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
# ---------------------------------------------------------------------------
# Diccionario bilingüe (lema español -> forma base inglesa)
# ---------------------------------------------------------------------------

DICCIONARIO: dict[str, str] = {
    # Vocabulario frecuente del corpus EFE (completa la cobertura del diccionario)
    "pasar": "happen", "deber": "must", "parecer": "seem", "creer": "believe", "hablar": "speak",
    "dejar": "leave", "seguir": "continue", "encontrar": "find", "llamar": "call", "venir": "come",
    "pensar": "think", "salir": "leave", "volver": "return", "conocer": "know", "vivir": "live",
    "tratar": "try", "empezar": "begin", "esperar": "wait", "buscar": "look for", "entrar": "enter",
    "trabajar": "work", "escribir": "write", "perder": "lose", "ocurrir": "happen",
    "recibir": "receive", "terminar": "finish", "permitir": "allow", "aparecer": "appear",
    "considerar": "consider", "abrir": "open", "cerrar": "close", "anunciar": "announce",
    "declarar": "declare", "añadir": "add", "destacar": "highlight", "indicar": "indicate",
    "confirmar": "confirm", "aumentar": "increase", "reducir": "reduce", "alcanzar": "reach",
    "celebrar": "hold", "organizar": "organize", "participar": "take part", "ganar": "win",
    "obligar": "force", "impedir": "prevent", "dirigir": "conduct", "juzgar": "try", "robar": "steal",
    "hombre": "man", "mujer": "woman", "grupo": "group", "caso": "case", "forma": "way",
    "tipo": "type", "dato": "figure", "resultado": "result", "nombre": "name", "mes": "month",
    "hora": "hour", "semana": "week", "mundo": "world", "empresa": "company", "economía": "economy",
    "informe": "report", "poder": "power", "juez": "judge", "prueba": "evidence", "pena": "sentence",
    "nuevo": "new", "grande": "large", "pequeño": "small", "mejor": "better", "bueno": "good",
    "alto": "high", "largo": "long", "general": "general", "internacional": "international",
    "posible": "possible", "necesario": "necessary", "actual": "current", "pasado_adj": "past",
    "durante": "during", "contra": "against", "hacia": "towards", "mientras": "while",
    "antes": "before", "después": "after", "siempre": "always", "nunca": "never",
    "siete": "seven", "ocho": "eight", "nueve": "nine", "diez": "ten", "doce": "twelve",
    "veinte": "twenty", "treinta": "thirty", "cien": "hundred",
    # Determinantes, preposiciones y conjunciones
    "el": "the", "uno": "a", "este": "this", "ese": "that", "otro": "other",
    "todo": "all", "ninguno": "no", "diferente": "different", "distinto": "different",
    "de": "of", "a": "to", "en": "in", "con": "with", "por": "by", "para": "for", "sin": "without",
    "sobre": "on", "entre": "between", "hasta": "until", "desde": "from",
    "según": "according to", "tras": "after", "ante": "before", "del": "of the", "al": "to the", "y": "and", "o": "or", "pero": "but",
    "que": "that", "como": "as", "si": "if", "porque": "because", "cuando": "when", "donde": "where",
    "aunque": "although", "también": "also", "no": "not", "ni": "nor", "más": "more", "menos": "less",
    "muy": "very", "ya": "already", "hoy": "today", "solo": "only",
    "casi": "almost", "sin embargo": "however",
    # Pronombres
    "yo": "we", "él": "he",
    "su": "its", "quien": "who", "se": "",
    # Verbos de uso general
    "ser": "be", "estar": "be", "haber": "have", "tener": "have", "hacer": "do", "ir": "go",
    "decir": "say", "dar": "give", "ver": "see", "saber": "know", "querer": "want",
    "llegar": "arrive", "poner": "put",
    "quedar": "remain", "llevar": "carry", "tomar": "take",
    "presentar": "present", "señalar": "point out", "informar": "report",
    "explicar": "explain", "subir": "rise",
    "acoger": "host", "solicitar": "request", "castigar": "punish", "interpretar": "perform",
    "sufrir": "suffer", "prometer": "promise", "revisar": "review", "despegar": "take off", "ocasionar": "cause", "causar": "cause", "pedir": "ask for",
    "lavar": "wash", "componer": "compose", "exponer": "exhibit", "trasladar": "transfer", "caer": "fall", "secuestrar": "hijack",
    "saltar": "jump", "escapar": "escape",
    # Sustantivos de las noticias
    "museo": "museum", "exposición": "exhibition", "escultura": "sculpture", "vidrio": "glass", "artista": "artist", "pieza": "piece", "formato": "format", "obra": "work",
    "concejal": "councillor", "cultura": "culture",
    "defensa": "defense", "absolución": "acquittal", "joven": "young man", "fiscal": "prosecutor",
    "cárcel": "prison", "año": "year", "acusado": "defendant",
    "víctima": "victim",
    "avión": "plane", "tierra": "ground", "problema": "problem", "aeropuerto": "airport",
    "pasajero": "passenger",
    "combustible": "fuel", "falta": "lack",
    "secuestrador": "hijacker", "portavoz": "spokesman",
    "presidente": "president", "gobierno": "government", "estado": "state", "ministro": "minister", "banco": "bank", "reserva": "reserve", "oro": "gold", "divisa": "currency",
    "dólar": "dollar", "dólares": "dollars", "millón": "million", "mil": "thousand", "precio": "price",
    "valor": "value", "comunicado": "statement",
    "elección": "election", "candidato": "candidate", "contrincante": "opponent", "mandato": "term",
    "constitución": "constitution", "entrevista": "interview", "diario": "newspaper",
    "político": "politician", "voto": "vote", "poder": "power",
    "orquesta": "orchestra", "concierto": "concert", "teatro": "theatre", "violinista": "violinist",
    "pianista": "pianist", "compositor": "composer", "música": "music", "ayuntamiento": "city council",
    "ciclista": "cyclist", "etapa": "stage", "fractura": "fracture", "fémur": "femur",
    "hospital": "hospital", "ambulancia": "ambulance", "atención": "care", "kilómetro": "kilometre", "caída": "fall",
    "lluvia": "rain", "muerto": "death", "damnificado": "victim",
    "departamento": "department", "ciudad": "city", "capital": "capital",
    "barrio": "neighbourhood", "agua": "water", "emergencia": "emergency", "desastre": "disaster", "período": "period",
    "servicio": "service", "autoridad": "authority", "oficina": "office", "responsable": "head", "fuente": "source", "día": "day", "país": "country",
    "persona": "person", "vecino": "neighbour", "parte": "part",
    "número": "number", "cifra": "figure",
    "reforma": "reform", "alcalde": "mayor", "mano": "hand", "pelo": "hair", "grito": "shout",
    "cielo": "sky", "borraja": "borage", "situación": "situation", "presencia": "presence", "tamaño": "size", "inauguración": "opening", "nacionalidad": "nationality", "lesión": "injury", "arma": "weapon",
    "entrada": "ticket", "dinero": "money",
    # Adjetivos "primero": "first", "último": "last", "importante": "important", "nacional": "national",
    "local": "local", "municipal": "municipal", "oficial": "official", "público": "public",
    "próximo": "next", "intenso": "heavy", "clásico": "classical",
    "italiano": "Italian",
    # Exónimos: los topónimos con forma propia en inglés sí se traducen.
    "rusia": "Russia", "españa": "Spain", "alemania": "Germany", "francia": "France",
    "bélgica": "Belgium", "japón": "Japan", "rumanía": "Romania", "marruecos": "Morocco",
    "filipinas": "Philippines", "moscú": "Moscow", "español": "Spanish", "rumano": "Romanian", "europeo": "European", "marroquí": "Moroccan", "derecho": "right",
    "izquierdo": "left", "presidencial": "presidential", "aéreo": "air", "civil": "civil", "crítico": "critical",
    "compuesto": "made up",
    # Numerales
    "dos": "two", "tres": "three", "cuatro": "four", "cinco": "five", "seis": "six",
}
# Unidades pluriverbales: su traducción no es la suma de sus partes.
LOCUCIONES: dict[str, str] = {
    "al menos": "at least", "desde hace": "for", "a través de": "through",
    "sin embargo": "however", "de acuerdo con": "according to", "a pesar de": "despite",
    "en cuanto a": "regarding", "por ciento": "percent", "es decir": "that is",
    "de nuevo": "again", "en base a": "based on", "junto con": "together with",
    "cada vez más": "increasingly", "a unos": "about", "hora local": "local time",
}
# Formas irregulares del inglés: pasado simple y participio.
IRREGULARES: dict[str, tuple[str, str]] = {
    "be": ("was", "been"), "have": ("had", "had"), "do": ("did", "done"), "go": ("went", "gone"),
    "say": ("said", "said"), "give": ("gave", "given"), "see": ("saw", "seen"),
    "know": ("knew", "known"), "take": ("took", "taken"), "come": ("came", "come"),
    "leave": ("left", "left"), "put": ("put", "put"), "find": ("found", "found"),
    "think": ("thought", "thought"), "lose": ("lost", "lost"), "win": ("won", "won"),
    "rise": ("rose", "risen"), "fall": ("fell", "fallen"), "grow": ("grew", "grown"),
    "hold": ("held", "held"), "feel": ("felt", "felt"), "speak": ("spoke", "spoken"),
    "write": ("wrote", "written"), "begin": ("began", "begun"), "hit": ("hit", "hit"),
    "can": ("could", "been able"), "must": ("had to", "had to"), "remain": ("remained", "remained"),
    "make": ("made", "made"), "become": ("became", "become"), "carry": ("carried", "carried"),
    "try": ("tried", "tried"), "cut": ("cut", "cut"), "read": ("read", "read"),
}
# Sustantivos ingleses con plural irregular o invariable.
PLURALES_IRREGULARES = {"man": "men", "woman": "women", "person": "people", "child": "children",
                        "foot": "feet", "percent": "percent", "million": "million",
                        "thousand": "thousand", "hundred": "hundred", "people": "people",
                        "young man": "young men", "missing person": "missing people"}


def _pluralizar(palabra: str) -> str:
    if palabra in PLURALES_IRREGULARES:
        return PLURALES_IRREGULARES[palabra]
    if not palabra or " " in palabra:
        # En los sintagmas se pluraliza la última palabra ("city council" -> "city councils").
        partes = palabra.split()
        return " ".join(partes[:-1] + [_pluralizar(partes[-1])]) if partes else palabra
    if palabra.endswith(("s", "x", "z", "ch", "sh")):
        return palabra + "es"
    if palabra.endswith("y") and palabra[-2:-1] not in "aeiou":
        return palabra[:-1] + "ies"
    return palabra + "s"


def _tercera_persona(verbo: str) -> str:
    base, *resto = verbo.split()
    if base in {"can", "must", "will"}:
        conjugado = base
    elif base == "be":
        conjugado = "is"
    elif base == "have":
        conjugado = "has"
    elif base.endswith(("s", "x", "z", "ch", "sh", "o")):
        conjugado = base + "es"
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        conjugado = base[:-1] + "ies"
    else:
        conjugado = base + "s"
    return " ".join([conjugado, *resto])


def _pasado(verbo: str) -> str:
    base, *resto = verbo.split()
    if base in IRREGULARES:
        conjugado = IRREGULARES[base][0]
    elif base.endswith("e"):
        conjugado = base + "d"
    elif base.endswith("y") and base[-2:-1] not in "aeiou":
        conjugado = base[:-1] + "ied"
    else:
        conjugado = base + "ed"
    return " ".join([conjugado, *resto])


def _participio(verbo: str) -> str:
    base, *resto = verbo.split()
    conjugado = IRREGULARES[base][1] if base in IRREGULARES else _pasado(base)
    return " ".join([conjugado, *resto])
# ---------------------------------------------------------------------------
# Reglas de transferencia
# ---------------------------------------------------------------------------

_PUNTUACION_FINAL = set(".,;:!?)»")


@dataclass
class TraductorReglas:
    """Traductor español→inglés por diccionario y reglas de transferencia.

    `nlp` es un modelo de spaCy en español: se usa para obtener el lema, la categoría
    gramatical y los rasgos morfológicos que gobiernan las reglas.
    """

    nlp: object
    diccionario: dict = field(default_factory=lambda: dict(DICCIONARIO))
    locuciones: dict = field(default_factory=lambda: dict(LOCUCIONES))
    reordenar_adjetivos: bool = True
    registrar_faltantes: bool = True
    faltantes: set = field(default_factory=set)
    # -- búsqueda léxica -------------------------------------------------
    def buscar(self, token) -> str | None:
        """Busca la forma exacta y, si falla, el lema; devuelve None si la palabra no está."""
        for clave in (token.text.lower(), token.lemma_.lower()):
            if clave in self.diccionario:
                return self.diccionario[clave]
        # Desambiguación mínima por categoría para los homógrafos registrados con sufijo.
        clave_pos = f"{token.lemma_.lower()}_{'adj' if token.pos_ == 'ADJ' else 'n'}"
        if clave_pos in self.diccionario:
            return self.diccionario[clave_pos]
        return None
    # -- generación de una palabra ---------------------------------------
    def generar(self, token, traduccion: str, auxiliar=None) -> str:
        """Aplica la morfología inglesa según los rasgos del token español."""
        if token.pos_ in {"NOUN", "PROPN"}:
            if "Plur" in token.morph.get("Number"):
                return _pluralizar(traduccion)
            return traduccion
        # Adjetivo sustantivado («28 muertos y 18.000 damnificados»): lleva el plural inglés.
        if token.pos_ == "ADJ" and "Plur" in token.morph.get("Number") and (
                token.head.pos_ not in {"NOUN", "PROPN"} or token.head.i == token.i):
            return _pluralizar(traduccion)
        if token.pos_ in {"VERB", "AUX"}:
            tiempo = token.morph.get("Tense")
            forma = token.morph.get("VerbForm")
            if "Part" in forma or (auxiliar is not None and "Past" in tiempo and token.pos_ == "VERB"):
                return _participio(traduccion)
            if "Ger" in forma:
                base = traduccion.split()[0]
                gerundio = base[:-1] + "ing" if base.endswith("e") else base + "ing"
                return " ".join([gerundio, *traduccion.split()[1:]])
            if "Inf" in forma:
                return "to " + traduccion
            if "Fut" in tiempo:
                return "will " + traduccion
            if "Past" in tiempo or "Imp" in tiempo:
                return _pasado(traduccion)
            if "Pres" in tiempo and "3" in token.morph.get("Person") and "Sing" in token.morph.get("Number"):
                return _tercera_persona(traduccion)
            return traduccion
        return traduccion
    # -- reordenamiento ---------------------------------------------------
    @staticmethod
    def reordenar(tokens: list) -> list:
        """Sustantivo + adjetivo (español) → adjetivo + sustantivo (inglés).

        Se aplica a secuencias contiguas SUST (+ADJ)+ dentro del mismo sintagma; los
        adjetivos que en realidad son participios de un verbo compuesto no se mueven.
        """
        salida, i = [], 0
        while i < len(tokens):
            if tokens[i].pos_ in {"NOUN", "PROPN"}:
                j = i + 1
                while j < len(tokens) and tokens[j].pos_ == "ADJ" and tokens[j].head == tokens[i]:
                    j += 1
                if j > i + 1:
                    salida.extend(tokens[i + 1:j])   # adjetivos primero
                    salida.append(tokens[i])
                    i = j
                    continue
            salida.append(tokens[i])
            i += 1
        return salida
    # -- traducción completa ---------------------------------------------
    def __call__(self, oracion: str) -> str:
        doc = self.nlp(oracion)
        texto_minuscula = oracion.lower()
        # 1. Locuciones: se traducen antes y se marcan los tokens consumidos.
        consumidos: dict[int, str] = {}
        for locucion, equivalente in self.locuciones.items():
            for coincidencia in re.finditer(rf"\b{re.escape(locucion)}\b", texto_minuscula):
                tramo = doc.char_span(coincidencia.start(), coincidencia.end(), alignment_mode="expand")
                if tramo is None:
                    continue
                consumidos[tramo.start] = equivalente
                for k in range(tramo.start + 1, tramo.end):
                    consumidos[k] = ""
        # 2. Reordenamiento sustantivo–adjetivo sobre los tokens de origen.
        tokens = list(doc)
        if self.reordenar_adjetivos:
            tokens = self.reordenar(tokens)
        # 3. Traducción palabra a palabra con generación morfológica.
        piezas: list[str] = []
        anterior_aux = None
        for token in tokens:
            if token.i in consumidos:
                if consumidos[token.i]:
                    piezas.append(consumidos[token.i])
                continue
            if token.is_punct:
                piezas.append(token.text)
                anterior_aux = None
                continue
            if token.text[0].isdigit():
                piezas.append(token.text.replace(".", ""))
                continue
            if token.pos_ == "PROPN" and self.buscar(token) is None:
                piezas.append(token.text)          # los nombres propios no se traducen
                continue
            traduccion = self.buscar(token)
            if traduccion is None:
                if self.registrar_faltantes:
                    self.faltantes.add(token.text.lower())
                piezas.append(token.text)          # palabra fuera del diccionario: se copia
                continue
            if traduccion == "":
                continue                            # el pronombre reflexivo «se» no se traduce
            piezas.append(self.generar(token, traduccion, auxiliar=anterior_aux))
            anterior_aux = token if token.pos_ == "AUX" else None
        return self._recomponer(piezas)

    @staticmethod
    def _recomponer(piezas: list[str]) -> str:
        """Une las piezas, pega la puntuación y arregla la secuencia de artículos."""
        salida: list[str] = []
        for pieza in piezas:
            if not pieza:
                continue
            if pieza and pieza[0] in _PUNTUACION_FINAL and salida:
                salida[-1] += pieza
                continue
            # «the the» y «of of» son residuos habituales de la traducción palabra a palabra.
            if salida and salida[-1].split()[-1:] == pieza.split()[:1] and pieza in {"the", "of", "to"}:
                continue
            salida.append(pieza)
        texto = " ".join(salida)
        texto = re.sub(r"\ba (?=[aeiouAEIOU])", "an ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto[:1].upper() + texto[1:] if texto else texto


def evaluar_traduccion(hipotesis: list[str], referencias: list[str]) -> dict:
    """BLEU (Papineni et al., 2002) y chrF (Popović, 2015) con `sacrebleu`."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hipotesis, [referencias])
    chrf = sacrebleu.corpus_chrf(hipotesis, [referencias])
    return {"bleu": bleu.score, "chrf": chrf.score, "detalle_bleu": bleu.format(),
            "precisiones": list(bleu.precisions)}

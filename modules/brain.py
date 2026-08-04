"""
modules/brain.py
────────────────
Procesador central de comandos (Brain).

Recibe el texto del comando (ya sin la wake word) y devuelve una respuesta.
Diseñado para crecer: los comandos se registran en un dict de handlers y pueden
ampliarse fácilmente sin tocar la lógica de despacho.

Compatibilidad: Linux, Android/Termux, cualquier Python 3.10+.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# ─── Tokens de salida ─────────────────────────────────────────────────────────

EXIT_TOKENS: frozenset[str] = frozenset({
    "salir", "adiós", "adios", "apágate", "apagate", "terminar", "hasta luego",
})

# ─── Respuestas de chiste ─────────────────────────────────────────────────────

JOKES: list[str] = [
    "¿Por qué los programadores prefieren el frío? Porque odian los bugs y en el calor aparecen más.",
    "¿Qué le dijo un bit al otro? Nos vemos en el bus de datos.",
    "Tengo un chiste sobre UDP, pero puede que no te llegue.",
    "Hay 10 tipos de personas en el mundo: las que entienden binario y las que no.",
    "¿Por qué el café estaba tan seguro de sí mismo? Porque tenía mucho espresso.",
    "Un SQL entra a un bar, ve dos mesas y pregunta: ¿puedo hacer un JOIN?",
    "Fui a una pelea entre dos funciones recursivas. Tardó una eternidad en terminar.",
    "¿Cómo organizan una fiesta los programadores? Con un commit y muchos push.",
]


# ─── Clase principal ──────────────────────────────────────────────────────────


class Brain:
    """Procesador central de comandos.

    Mapea palabras clave presentes en el texto del comando a funciones handler.
    Diseñado para ser extensible: se pueden agregar comandos con `register_command`.

    Attributes:
        _commands: Diccionario {keyword: handler_fn} para el despacho de comandos.
    """

    def __init__(self) -> None:
        self._commands: Dict[str, Callable[[], str]] = {
            "hora":            self._cmd_time,
            "qué hora":        self._cmd_time,
            "que hora":        self._cmd_time,
            "chiste":          self._cmd_joke,
            "cuenta un chiste": self._cmd_joke,
            "dime un chiste":  self._cmd_joke,
            "ayuda":           self._cmd_help,
            "qué sabes hacer": self._cmd_help,
            "que sabes hacer": self._cmd_help,
            "comandos":        self._cmd_help,
        }
        logger.debug("Brain inicializado con %d comandos.", len(self._commands))

    # ── Registro de comandos ──────────────────────────────────────────────────

    def register_command(self, keyword: str, handler: Callable[[], str]) -> None:
        """Registra un nuevo comando en tiempo de ejecución.

        Permite agregar funcionalidades (plugins, herramientas) sin modificar
        el código del Brain.

        Args:
            keyword: Palabra o frase clave que dispara el handler.
            handler: Función sin argumentos que retorna la respuesta como str.
        """
        self._commands[keyword] = handler
        logger.debug("Comando registrado: '%s'", keyword)

    # ── Handlers de comandos ──────────────────────────────────────────────────

    @staticmethod
    def _cmd_time() -> str:
        hora = datetime.now().strftime("%H:%M")
        return f"Son las {hora}."

    @staticmethod
    def _cmd_joke() -> str:
        return random.choice(JOKES)

    @staticmethod
    def _cmd_help() -> str:
        return (
            "Puedo decirte la hora, contar un chiste, "
            "o despedirme cuando me digas adiós."
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def is_exit_command(self, text: str) -> bool:
        """Devuelve True si el texto contiene un token de salida."""
        if not text:
            return False
        normalized = text.lower().strip()
        return any(token in normalized for token in EXIT_TOKENS)

    def handle(self, text: str) -> str:
        """Despacha el texto a un handler y retorna la respuesta.

        Args:
            text: Comando del usuario (sin la wake word, ya normalizado).

        Returns:
            Respuesta en formato texto que será vocalizada por el TTS.
        """
        if not text:
            return "No te he entendido, ¿puedes repetirlo?"

        normalized = text.lower().strip()
        logger.info("Procesando comando: '%s'", normalized)

        for keyword, handler in self._commands.items():
            if keyword in normalized:
                response = handler()
                logger.info("Respuesta: '%s'", response)
                return response

        logger.info("Comando desconocido: '%s'", normalized)
        return (
            f"No conozco el comando '{text}'. "
            "Di 'ayuda' para saber qué puedo hacer."
        )

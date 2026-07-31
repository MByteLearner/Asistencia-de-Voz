import random
from datetime import datetime
from typing import Callable, Dict


EXIT_TOKENS = {"salir", "adiós", "adios", "apágate", "apagate", "terminar", "hasta luego"}


JOKES = [
    "¿Por qué los programadores prefieren el frío? Porque odian los bugs y en el calor aparecen más.",
    "¿Qué le dijo un bit al otro? Nos vemos en el bus de datos.",
    "Tengo un chiste sobre UDP, pero puede que no te llegue.",
    "Hay 10 tipos de personas en el mundo: las que entienden binario y las que no.",
    "¿Por qué el café estaba tan seguro de sí mismo? Porque tenía mucho espresso.",
    "Un SQL entra a un bar, ve dos mesas y pregunta: ¿puedo hacer un JOIN?",
    "Fui a una pelea entre dos funciones recursivas. Tardó una eternidad en terminar.",
    "¿Cómo organizan una fiesta los programadores? Con un commit y muchos push.",
]


class Brain:
    def __init__(self):
        self._commands: Dict[str, Callable[[], str]] = {
            "hora": self._cmd_time,
            "qué hora": self._cmd_time,
            "que hora": self._cmd_time,
            "chiste": self._cmd_joke,
            "cuenta un chiste": self._cmd_joke,
            "dime un chiste": self._cmd_joke,
            "ayuda": self._cmd_help,
            "qué sabes hacer": self._cmd_help,
            "que sabes hacer": self._cmd_help,
            "comandos": self._cmd_help,
        }

    @staticmethod
    def _cmd_time() -> str:
        now = datetime.now()
        hora = now.strftime("%H:%M")
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

    def is_exit_command(self, text: str) -> bool:
        if not text:
            return False
        normalized = text.lower().strip()
        return any(token in normalized for token in EXIT_TOKENS)

    def handle(self, text: str) -> str:
        if not text:
            return "No te he entendido, ¿puedes repetirlo?"

        normalized = text.lower().strip()

        for keyword, handler in self._commands.items():
            if keyword in normalized:
                return handler()

        return (
            f"No conozco el comando '{text}'. "
            "Di 'ayuda' para saber qué puedo hacer."
        )

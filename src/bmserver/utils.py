import shutil
from os import terminal_size

from rich.console import Console


def get_console() -> Console:
    size: terminal_size = shutil.get_terminal_size()
    return Console(width=size.columns)

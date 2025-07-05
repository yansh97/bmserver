import typer
from rich.console import Console

from bmserver.chat.cli import app as chat_app
from bmserver.schema import Environment
from bmserver.utils import get_console

app = typer.Typer(no_args_is_help=True, help="BigModel Server CLI Toolkit")
app.add_typer(typer_instance=chat_app, name="chat")


@app.command()
def env() -> None:
    """Detect environment."""
    environment: Environment = Environment.detect()
    console: Console = get_console()
    console.print(environment)

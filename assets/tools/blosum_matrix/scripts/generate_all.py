#!/usr/bin/env python3
"""Regenerate distributable substitution-matrix resources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

_SCRIPTS = Path(__file__).resolve().parent

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main() -> None:
    """Regenerate resources that do not require an external source dataset."""

    for script in ("export_blosum62.py",):
        path = _SCRIPTS / script
        typer.echo(f"==> {path.name}")
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    app()

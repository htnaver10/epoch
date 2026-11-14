"""Portable entry point. Paths are relative to this full-resolution folder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from firefullres.cli import main

if __name__ == '__main__':
    main()

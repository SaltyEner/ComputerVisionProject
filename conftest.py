"""Make the project root importable so `from src import ...` works under pytest.

When CI runs bare ``pytest`` (instead of ``python -m pytest``) the current
directory is not on ``sys.path``; this conftest puts the repo root there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

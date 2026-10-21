"""attack-sequence-mining source package."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
for sub in ["core", "eda", "mining", "models", "eval"]:
    p = str(_SRC / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

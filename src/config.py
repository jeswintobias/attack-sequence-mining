"""
Compatibility layer: re-export everything from src.core.config.
Allows both `import config` and `from core import config`.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_CORE_DIR = _SRC_DIR / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import *
import core.config as _cfg

globals().update({k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")})

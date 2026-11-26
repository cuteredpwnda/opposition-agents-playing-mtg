# Compat shim — aliases src.engine.mana → src.engine_legacy.mana
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import mana as _real
_sys.modules[__name__] = _real

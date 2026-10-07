# Compat shim — aliases src.engine.abilities → src.engine_legacy.abilities
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import abilities as _real
_sys.modules[__name__] = _real

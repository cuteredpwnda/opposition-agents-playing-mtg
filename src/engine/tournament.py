# Compat shim — aliases src.engine.tournament → src.engine_legacy.tournament
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import tournament as _real
_sys.modules[__name__] = _real

# Compat shim — aliases src.engine.stack → src.engine_legacy.stack
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import stack as _real
_sys.modules[__name__] = _real

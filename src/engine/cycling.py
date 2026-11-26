# Compat shim — aliases src.engine.cycling → src.engine_legacy.cycling
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import cycling as _real
_sys.modules[__name__] = _real

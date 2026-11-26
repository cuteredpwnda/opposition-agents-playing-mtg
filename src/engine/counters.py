# Compat shim — aliases src.engine.counters → src.engine_legacy.counters
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import counters as _real
_sys.modules[__name__] = _real

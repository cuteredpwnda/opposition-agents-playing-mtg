# Compat shim — aliases src.engine.triggers → src.engine_legacy.triggers
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import triggers as _real
_sys.modules[__name__] = _real

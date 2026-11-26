# Compat shim — aliases src.engine.cascade → src.engine_legacy.cascade
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import cascade as _real
_sys.modules[__name__] = _real

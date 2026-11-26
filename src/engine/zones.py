# Compat shim — aliases src.engine.zones → src.engine_legacy.zones
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import zones as _real
_sys.modules[__name__] = _real

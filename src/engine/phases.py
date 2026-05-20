# Compat shim — aliases src.engine.phases → src.engine_legacy.phases
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import phases as _real
_sys.modules[__name__] = _real

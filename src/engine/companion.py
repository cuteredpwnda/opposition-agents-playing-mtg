# Compat shim — aliases src.engine.companion → src.engine_legacy.companion
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import companion as _real
_sys.modules[__name__] = _real

# Compat shim — aliases src.engine.combat → src.engine_legacy.combat
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import combat as _real
_sys.modules[__name__] = _real

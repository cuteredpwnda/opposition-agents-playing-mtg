# Compat shim — aliases src.engine.replacement_effects → src.engine_legacy.replacement_effects
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import replacement_effects as _real
_sys.modules[__name__] = _real

# Compat shim — aliases src.engine.continuous_effects → src.engine_legacy.continuous_effects
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import continuous_effects as _real
_sys.modules[__name__] = _real

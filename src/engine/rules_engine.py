# Compat shim — aliases src.engine.rules_engine → src.engine_legacy.rules_engine
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import rules_engine as _real
_sys.modules[__name__] = _real

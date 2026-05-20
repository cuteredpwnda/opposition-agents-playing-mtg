# Compat shim — aliases src.engine.watchers → src.engine_legacy.watchers
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import watchers as _real
_sys.modules[__name__] = _real

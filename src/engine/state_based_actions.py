# Compat shim — aliases src.engine.state_based_actions → src.engine_legacy.state_based_actions
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import state_based_actions as _real
_sys.modules[__name__] = _real

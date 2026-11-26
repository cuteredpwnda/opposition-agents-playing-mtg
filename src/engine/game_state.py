# Compat shim — aliases src.engine.game_state → src.engine_legacy.game_state
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import game_state as _real
_sys.modules[__name__] = _real

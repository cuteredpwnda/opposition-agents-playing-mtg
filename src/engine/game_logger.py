# Compat shim — aliases src.engine.game_logger → src.engine_legacy.game_logger
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import game_logger as _real
_sys.modules[__name__] = _real

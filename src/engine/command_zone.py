# Compat shim — aliases src.engine.command_zone → src.engine_legacy.command_zone
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import command_zone as _real
_sys.modules[__name__] = _real

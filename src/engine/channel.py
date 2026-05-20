# Compat shim — aliases src.engine.channel → src.engine_legacy.channel
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import channel as _real
_sys.modules[__name__] = _real

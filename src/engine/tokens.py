# Compat shim — aliases src.engine.tokens → src.engine_legacy.tokens
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import tokens as _real
_sys.modules[__name__] = _real

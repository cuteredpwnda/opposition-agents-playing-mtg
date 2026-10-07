# Compat shim — aliases src.engine.static_abilities → src.engine_legacy.static_abilities
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import static_abilities as _real
_sys.modules[__name__] = _real

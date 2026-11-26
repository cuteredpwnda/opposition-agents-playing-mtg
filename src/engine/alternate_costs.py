# Compat shim — aliases src.engine.alternate_costs → src.engine_legacy.alternate_costs
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import alternate_costs as _real
_sys.modules[__name__] = _real

# Compat shim — aliases src.engine.card_database → src.engine_legacy.card_database
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import card_database as _real
_sys.modules[__name__] = _real

# Compat shim — aliases src.engine.spell_effects → src.engine_legacy.spell_effects
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import spell_effects as _real
_sys.modules[__name__] = _real

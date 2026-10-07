# Compat shim — aliases src.engine.llm_agent → src.engine_legacy.llm_agent
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import llm_agent as _real
_sys.modules[__name__] = _real

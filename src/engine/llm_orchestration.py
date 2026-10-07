# Compat shim — aliases src.engine.llm_orchestration → src.engine_legacy.llm_orchestration
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import llm_orchestration as _real
_sys.modules[__name__] = _real

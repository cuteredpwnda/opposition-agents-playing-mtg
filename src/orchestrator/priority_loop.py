# Compat shim — aliases src.orchestrator.priority_loop → src.orchestrator_legacy.priority_loop
import sys as _sys
from src.orchestrator_legacy import priority_loop as _real
_sys.modules[__name__] = _real

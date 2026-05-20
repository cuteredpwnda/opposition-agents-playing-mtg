# Compat shim — aliases src.orchestrator.game_runner → src.orchestrator_legacy.game_runner
import sys as _sys
from src.orchestrator_legacy import game_runner as _real
_sys.modules[__name__] = _real

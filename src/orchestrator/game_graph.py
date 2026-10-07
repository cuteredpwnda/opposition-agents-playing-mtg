# Compat shim — aliases src.orchestrator.game_graph → src.orchestrator_legacy.game_graph
import sys as _sys
from src.orchestrator_legacy import game_graph as _real
_sys.modules[__name__] = _real

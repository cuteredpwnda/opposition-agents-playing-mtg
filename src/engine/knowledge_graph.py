# Compat shim — aliases src.engine.knowledge_graph → src.engine_legacy.knowledge_graph
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import knowledge_graph as _real
_sys.modules[__name__] = _real

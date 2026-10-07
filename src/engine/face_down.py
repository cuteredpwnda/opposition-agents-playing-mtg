# Compat shim — aliases src.engine.face_down → src.engine_legacy.face_down
# Allows old import paths to keep working after Phase 2 reorganization.
import sys as _sys
from src.engine_legacy import face_down as _real
_sys.modules[__name__] = _real

"""
BGBKD — Battery Graph Bistable Koopman Dictionary
══════════════════════════════════════════════════════════════
Pack-Level Thermal Runaway Cascade Forecasting.
226 seconds lead time before voltage trigger.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from .eis.state        import EISState, eis_to_u
from .core.observables import phi3, phi4, psi_score, BGBKDObservables
from .core.edmd        import BGBKDEDMDSolver
from .core.cascade     import CascadeReport, TwoHopCascadeDetector, PackFragility
from .core.estimator   import BGBKDAdaptiveEstimator
from .infrastructure.synthetic_adapter import simulate_eis, simulate_pack
from .infrastructure.config_loader     import load_bgbkd

BGBKDConfig = dict  # config loaded via load_bgbkd()
__version__ = "0.1.0"

__all__ = [
    "EISState", "eis_to_u",
    "phi3", "phi4", "psi_score",
    "BGBKDObservables",
    "BGBKDEDMDSolver",
    "CascadeReport", "TwoHopCascadeDetector", "PackFragility",
    "BGBKDAdaptiveEstimator",
    "simulate_eis", "simulate_pack",
    "load_bgbkd", "BGBKDConfig",
    "__version__",
]

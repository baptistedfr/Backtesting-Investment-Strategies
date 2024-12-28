# Version du package
__version__ = "0.1.0"

# Import des modules principaux
from .backtester import Backtester
from .results import Results
from .data_input import DataInput


__all__ = [
    # Modules principaux
    "Backtester",
    "DataInput",
    "Results",
] 
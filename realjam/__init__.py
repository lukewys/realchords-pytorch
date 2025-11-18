"""RealJam - Lightweight real-time music accompaniment system."""

__version__ = "1.0.0"

from realjam.server import main as start_server

__all__ = ["start_server", "__version__"]

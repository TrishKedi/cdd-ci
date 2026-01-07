"""Command-line interface components for code duplication detection.

This package provides CLI tools and engines for running duplicate
code detection from the command line.
"""

from .duplication_detection_engine import DuplicationDetectionEngine

__all__ = ["DuplicationDetectionEngine"]
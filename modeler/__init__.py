"""Optional wireframe visualization of an OptimizationResult.

This package is deliberately one-way: it imports frame_optimizer, but nothing
in frame_optimizer imports it. The only caller is a guarded `try: import`
in main.py, so deleting this folder disables visualization without breaking
the optimizer, the tests, or the CSV output.

Requires plotly (the core package does not).
"""
from .wireframe import visualize_result

__all__ = ["visualize_result"]

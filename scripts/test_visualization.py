#!/usr/bin/env python3
"""Test the visualization with string grid values"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization.layered_visualizer import LayeredVisualizer
from src.nlp.base_generator_nlp import BaseGeneratorNLP


def test_visualization():
    """Test visualization with generated grid"""

    print("1. Generating a test base...")
    nlp = BaseGeneratorNLP()
    grid, desc = nlp.generate_base("Create a base for 5 colonists", 30, 30)

    print(f"2. Grid generated: shape={grid.shape}, dtype={grid.dtype}")
    print(f"   Sample values: {grid[10, 10]}, {grid[15, 15]}, {grid[20, 20]}")

    print("3. Creating visualization...")
    visualizer = LayeredVisualizer()
    try:
        img = visualizer.visualize(
            grid, title="Test Base", show_legend=True, flip_y=False
        )
        img.save("test_visualization.png")
        print("   ✓ Visualization created successfully: test_visualization.png")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()

#!/usr/bin/env python3
"""Generate simple example bases using NLP"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.visualization.base_visualizer import BaseVisualizer

def main():
    """Generate various example bases"""

    # Create visualizer
    visualizer = BaseVisualizer(scale=8)

    # Create NLP generator (without save file for simplicity)
    nlp = BaseGeneratorNLP()

    examples = [
        ("Compact defensive base for 6 colonists with killbox", 50, 50, "example1_defensive.png"),
        ("Spacious production base with 4 workshops and large storage", 70, 70, "example2_production.png"),
        ("Agricultural base for 8 colonists with storage and kitchen", 60, 60, "example3_agricultural.png"),
        ("Efficient base for 10 colonists with medical bay and research", 65, 65, "example4_medical.png"),
        ("Comfortable base with recreation room for 7 colonists", 55, 55, "example5_comfort.png"),
    ]

    for i, (request, width, height, filename) in enumerate(examples, 1):
        print(f"\n{i}. Generating: {request[:50]}...")

        # Generate base
        grid, description = nlp.generate_base(request, width, height)

        # Save visualization
        output_path = f"output/{filename}"
        visualizer.visualize(grid, output_path, title=request[:40])

        # Print summary
        print(f"   [OK] Saved to {output_path}")
        for line in description.split('\n')[:3]:
            if line.strip():
                print(f"   {line}")

    print("\n[SUCCESS] All examples generated successfully!")
    print("Check output/ folder for PNG files")

if __name__ == "__main__":
    main()

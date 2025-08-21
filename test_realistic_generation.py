#!/usr/bin/env python3
"""Test realistic base generation with actual furniture and layouts"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.generators.realistic_base_generator import RealisticBaseGenerator
from src.visualization.realistic_visualizer import RealisticBaseVisualizer


def main():
    """Generate various realistic example bases"""
    
    # Create visualizer
    visualizer = RealisticBaseVisualizer(scale=8)
    
    # Test examples with different configurations
    examples = [
        {
            "name": "Small Starter Base",
            "params": {
                "num_bedrooms": 3,
                "include_kitchen": True,
                "include_dining": True,
                "include_workshop": 1,
                "include_storage": True,
                "include_hospital": False
            },
            "size": (60, 60),
            "output": "output/realistic_starter.png"
        },
        {
            "name": "Medium Colony",
            "params": {
                "num_bedrooms": 6,
                "include_kitchen": True,
                "include_dining": True,
                "include_workshop": 2,
                "include_storage": True,
                "include_hospital": True
            },
            "size": (80, 80),
            "output": "output/realistic_medium.png"
        },
        {
            "name": "Large Production Base",
            "params": {
                "num_bedrooms": 8,
                "include_kitchen": True,
                "include_dining": True,
                "include_workshop": 4,
                "include_storage": True,
                "include_hospital": True
            },
            "size": (100, 100),
            "output": "output/realistic_large.png"
        },
        {
            "name": "Compact Efficient Base",
            "params": {
                "num_bedrooms": 4,
                "include_kitchen": True,
                "include_dining": False,  # Combined with kitchen
                "include_workshop": 2,
                "include_storage": True,
                "include_hospital": True
            },
            "size": (70, 70),
            "output": "output/realistic_compact.png"
        },
        {
            "name": "Hospital Complex",
            "params": {
                "num_bedrooms": 5,
                "include_kitchen": True,
                "include_dining": True,
                "include_workshop": 1,
                "include_storage": True,
                "include_hospital": True
            },
            "size": (75, 75),
            "output": "output/realistic_hospital.png"
        }
    ]
    
    print("=" * 60)
    print("REALISTIC RIMWORLD BASE GENERATOR")
    print("=" * 60)
    
    for i, config in enumerate(examples, 1):
        print(f"\n{i}. Generating: {config['name']}")
        print("-" * 40)
        
        # Create generator
        width, height = config["size"]
        generator = RealisticBaseGenerator(width, height)
        
        # Generate base
        grid = generator.generate_base(**config["params"])
        
        # Get description
        description = generator.get_description()
        print(description)
        
        # Save visualization
        visualizer.visualize(grid, config["output"], title=config["name"])
        print(f"[OK] Saved to {config['output']}")
        
        # Show ASCII preview (small section)
        print("\nASCII Preview (top-left 40x20):")
        print("-" * 40)
        preview = visualizer.visualize_ascii(grid[:20, :40])
        print(preview)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL BASES GENERATED SUCCESSFULLY!")
    print("Check output/ folder for visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main()
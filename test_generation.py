#!/usr/bin/env python3
"""Test generation functionality directly"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.generators.requirements_driven_generator import RequirementsDrivenGenerator


def test_generation():
    """Test the generation pipeline"""

    # Test parameters
    description = "Create a base for 10 colonists with bedrooms, kitchen, and workshop"
    width = 60
    height = 60

    print(f"Testing generation with: {width}x{height}")
    print(f"Description: {description}")

    # Test NLP parsing
    print("\n1. Testing NLP parsing...")
    try:
        nlp = BaseGeneratorNLP()
        requirements = nlp.parse_request(description)
        print(
            f"   ✓ Parsed: colonists={requirements.num_colonists}, defense={requirements.defense_level}"
        )
    except Exception as e:
        print(f"   ✗ NLP parsing failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Check for AlphaPrefabs
    print("\n2. Checking for AlphaPrefabs...")
    alpha_prefabs_path = None
    possible_paths = [
        Path("data/AlphaPrefabs"),
        Path("../AlphaPrefabs"),
        Path("../../AlphaPrefabs"),
    ]

    for path in possible_paths:
        if path.exists() and (path / "PrefabSets").exists():
            alpha_prefabs_path = path
            print(f"   ✓ Found at: {alpha_prefabs_path}")
            break

    if not alpha_prefabs_path:
        print("   ✗ AlphaPrefabs not found")

    # Test smart generation
    if alpha_prefabs_path:
        print("\n3. Testing RequirementsDrivenGenerator...")
        try:
            generator = RequirementsDrivenGenerator(width, height, alpha_prefabs_path)
            grid = generator.generate_from_requirements(requirements=requirements)
            if grid is not None:
                print(f"   ✓ Generated grid: shape={grid.shape}")
            else:
                print("   ✗ Generator returned None")
        except Exception as e:
            print(f"   ✗ Generation failed: {e}")
            import traceback

            traceback.print_exc()

    # Test NLP fallback
    print("\n4. Testing NLP fallback generation...")
    try:
        grid, desc = nlp.generate_base(description, width, height)
        if grid is not None:
            print(f"   ✓ Generated grid: shape={grid.shape}")
            print(f"   Description: {desc[:100]}...")
        else:
            print("   ✗ NLP generation returned None")
    except Exception as e:
        print(f"   ✗ NLP generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_generation()

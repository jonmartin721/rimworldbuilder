"""
Test hybrid generation with real prefabs and Claude AI integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
from src.ai.claude_base_designer import ClaudeBaseDesigner, BaseDesignRequest
from src.generators.wfc_generator import TileType


def test_hybrid_generation():
    """Test the hybrid prefab generator"""
    print("=" * 60)
    print("Hybrid Prefab Generation Test")
    print("=" * 60)

    # 1. Test with real prefabs as anchors
    print("\n1. Using Real Prefabs as Anchors")
    print("-" * 40)

    alpha_path = Path("data/AlphaPrefabs")
    if not alpha_path.exists():
        print("AlphaPrefabs not found. Please clone the repository first.")
        return

    # Create hybrid generator
    generator = HybridPrefabGenerator(
        width=60, height=60, alpha_prefabs_path=alpha_path
    )

    # Generate with specific prefab categories
    print("\nGenerating base with real prefab anchors...")
    grid = generator.generate_with_prefab_anchors(
        prefab_categories=["bedroom", "kitchen", "storage", "workshop"],
        num_prefabs=4,
        fill_with_wfc=True,
    )

    # Save visualization
    visualize_grid(grid, "hybrid_prefab_base.png")
    print("Saved visualization to hybrid_prefab_base.png")

    # Report on placed prefabs
    print(f"\nPlaced {len(generator.placed_prefabs)} real prefabs:")
    for prefab in generator.placed_prefabs:
        print(f"  - {prefab.layout.def_name} ({prefab.category}) at {prefab.position}")

    # 2. Test Claude AI Integration
    print("\n\n2. Claude AI Base Design")
    print("-" * 40)

    designer = ClaudeBaseDesigner()

    # Create design request
    request = BaseDesignRequest(
        colonist_count=6,
        map_size=(60, 60),
        biome="temperate_forest",
        difficulty="medium",
        priorities=["efficiency", "comfort"],
        special_requirements=["workshop", "recreation", "hospital"],
        available_space=(50, 50),
    )

    print("\nRequesting intelligent base design...")
    plan = designer.design_base(request)

    if plan:
        print("\n=== AI-Generated Base Plan ===")
        print(f"Strategy: {plan.layout_strategy}")
        print(f"Defense: {plan.defense_strategy}")
        print("\nRoom Layout:")
        for spec in plan.room_specs[:8]:  # Show first 8 rooms
            print(
                f"  {spec.quantity}x {spec.room_type}: {spec.size[0]}x{spec.size[1]} tiles"
            )
            if spec.adjacency_preferences:
                print(f"    Near: {', '.join(spec.adjacency_preferences)}")

        print("\nSpecial Considerations:")
        for consideration in plan.special_considerations[:3]:
            print(f"  - {consideration}")

    # 3. Combine both approaches
    print("\n\n3. AI-Guided Prefab Selection")
    print("-" * 40)

    if plan:
        # Use AI plan to select prefab categories
        prefab_categories = []
        for spec in plan.room_specs:
            if spec.room_type not in prefab_categories:
                prefab_categories.append(spec.room_type)
                if len(prefab_categories) >= 5:
                    break

        print(f"AI recommends prefabs for: {', '.join(prefab_categories)}")

        # Generate with AI-selected prefabs
        generator2 = HybridPrefabGenerator(
            width=60, height=60, alpha_prefabs_path=alpha_path
        )

        grid2 = generator2.generate_with_prefab_anchors(
            prefab_categories=prefab_categories,
            num_prefabs=len(prefab_categories),
            fill_with_wfc=True,
        )

        visualize_grid(grid2, "ai_guided_prefab_base.png")
        print("\nSaved AI-guided generation to ai_guided_prefab_base.png")

    print("\n" + "=" * 60)
    print("Hybrid generation test complete!")
    print("Benefits demonstrated:")
    print("  1. Real prefabs provide authentic RimWorld room designs")
    print("  2. WFC fills between prefabs intelligently")
    print("  3. Claude AI can guide prefab selection and placement")
    print("=" * 60)


def visualize_grid(grid: np.ndarray, filename: str):
    """Create visualization of generated grid"""
    scale = 8
    height, width = grid.shape
    img = Image.new("RGB", (width * scale, height * scale), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Color mapping - check what's actually in TileType
    colors = {}
    for tile_type in TileType:
        if tile_type == TileType.EMPTY:
            colors[tile_type] = (30, 30, 30)
        elif tile_type == TileType.WALL:
            colors[tile_type] = (150, 150, 150)
        elif tile_type == TileType.DOOR:
            colors[tile_type] = (139, 69, 19)
        elif tile_type == TileType.CORRIDOR:
            colors[tile_type] = (80, 80, 80)
        elif tile_type == TileType.BEDROOM:
            colors[tile_type] = (100, 150, 200)
        elif tile_type == TileType.KITCHEN:
            colors[tile_type] = (255, 140, 0)
        elif tile_type == TileType.STORAGE:
            colors[tile_type] = (0, 200, 0)
        elif tile_type == TileType.WORKSHOP:
            colors[tile_type] = (255, 215, 0)
        elif tile_type == TileType.RECREATION:
            colors[tile_type] = (218, 112, 214)
        else:
            # For any other types
            colors[tile_type] = (100, 100, 100)

    # Draw tiles
    for y in range(height):
        for x in range(width):
            try:
                tile_type = TileType(grid[y, x])
                color = colors.get(tile_type, (100, 100, 100))
            except (ValueError, KeyError):
                # Handle string values from prefabs
                if grid[y, x] == "empty":
                    color = (30, 30, 30)
                elif grid[y, x] == "wall":
                    color = (150, 150, 150)
                elif grid[y, x] == "door":
                    color = (139, 69, 19)
                else:
                    color = (100, 100, 100)

            px = x * scale
            py = y * scale

            # Draw with slight border for visibility
            draw.rectangle(
                [px, py, px + scale - 1, py + scale - 1],
                fill=color,
                outline=(20, 20, 20),
            )

    # Add grid lines for clarity
    for x in range(0, width * scale, scale * 10):
        draw.line([(x, 0), (x, height * scale)], fill=(50, 50, 50), width=1)
    for y in range(0, height * scale, scale * 10):
        draw.line([(0, y), (width * scale, y)], fill=(50, 50, 50), width=1)

    # Add title
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        draw.text((5, 5), "Hybrid Prefab Generation", fill=(255, 255, 255), font=font)
    except (IOError, OSError):
        pass

    img.save(filename)


if __name__ == "__main__":
    test_hybrid_generation()

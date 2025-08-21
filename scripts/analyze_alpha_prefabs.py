"""
Analyze real prefab designs from AlphaPrefabs mod.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from src.generators.alpha_prefab_parser import AlphaPrefabParser
from src.generators.prefab_analyzer import PrefabAnalyzer
import logging

logging.basicConfig(level=logging.INFO)


def analyze_alpha_prefabs():
    """Analyze real prefab designs from AlphaPrefabs mod"""

    # Path to the cloned AlphaPrefabs repository
    alpha_prefabs_path = Path("data/AlphaPrefabs")

    if not alpha_prefabs_path.exists():
        print(f"AlphaPrefabs mod not found at {alpha_prefabs_path}")
        return

    print("Parsing AlphaPrefabs mod layouts...")
    parser = AlphaPrefabParser(alpha_prefabs_path)

    # Parse all layouts
    all_layouts = parser.parse_all_layouts()
    print(f"Found {len(all_layouts)} layouts")

    # Convert to PrefabDesign format
    prefab_designs = []
    for layout in all_layouts[:20]:  # Analyze first 20 for now
        design = parser.convert_to_prefab_design(layout)
        prefab_designs.append(design)
        print(
            f"  Converted: {design.name} ({design.width}x{design.height}, category: {design.metadata['category']})"
        )

    # Analyze with PrefabAnalyzer
    analyzer = PrefabAnalyzer()

    print("\nAnalyzing prefab patterns...")
    for design in prefab_designs:
        try:
            analyzer.analyze_prefab(design)
            print(f"  Analyzed: {design.name}")
        except Exception as e:
            print(f"  Error analyzing {design.name}: {e}")

    # Get learned rules
    rules = analyzer.get_learned_rules()

    print("\n=== LEARNED PATTERNS FROM REAL PREFABS ===")

    print("\nRoom Size Statistics:")
    for room_type, stats in rules.get("room_sizes", {}).items():
        print(f"  Room Type {room_type}:")
        print(f"    Average size: {stats['mean']:.1f} tiles")
        print(f"    Range: {stats['min']}-{stats['max']} tiles")

    print("\nRoom Adjacency Preferences:")
    for room_type, neighbors in rules.get("adjacencies", {}).items():
        if neighbors:
            print(f"  Room Type {room_type} prefers to be next to:")
            for neighbor, probability in sorted(
                neighbors.items(), key=lambda x: x[1], reverse=True
            )[:3]:
                print(f"    Room Type {neighbor}: {probability:.2%}")

    print("\nRoom Shape Preferences:")
    for room_type, shape in rules.get("room_shapes", {}).items():
        print(
            f"  Room Type {room_type}: aspect ratio {shape['mean']:.2f} ± {shape['std']:.2f}"
        )

    # Save learned patterns
    patterns_file = Path("learned_patterns_alpha.json")
    analyzer.save_learned_patterns(patterns_file)
    print(f"\nSaved learned patterns to {patterns_file}")

    # Visualize some of the real prefabs
    print("\nVisualizing sample prefabs...")
    for i, design in enumerate(prefab_designs[:3]):
        visualize_prefab(design, f"alpha_prefab_{i}.png")
        print(f"  Saved {design.name} as alpha_prefab_{i}.png")


def visualize_prefab(design, filename):
    """Visualize a prefab design"""
    scale = 20
    img_width = design.width * scale
    img_height = design.height * scale

    img = Image.new("RGB", (img_width, img_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Color mapping
    colors = {
        0: (30, 30, 30),  # Empty
        1: (150, 150, 150),  # Wall
        2: (139, 69, 19),  # Door
        3: (100, 150, 200),  # Bedroom furniture
        4: (255, 140, 0),  # Kitchen/table
        5: (200, 100, 50),  # Dining/chairs
        6: (100, 200, 255),  # Climate control
        7: (255, 255, 100),  # Lighting
        8: (34, 139, 34),  # Decoration
        9: (200, 100, 200),  # Recreation/utility
        10: (150, 75, 0),  # Animal
    }

    # Draw tiles
    for y in range(design.height):
        for x in range(design.width):
            tile_type = design.layout[y, x]
            color = colors.get(tile_type, (100, 100, 100))

            px = x * scale
            py = y * scale

            if tile_type == 1:  # Wall
                draw.rectangle(
                    [px, py, px + scale - 1, py + scale - 1],
                    fill=color,
                    outline=(0, 0, 0),
                )
            elif tile_type == 2:  # Door
                draw.rectangle(
                    [px + scale // 4, py, px + 3 * scale // 4, py + scale - 1],
                    fill=color,
                    outline=(0, 0, 0),
                )
            elif tile_type > 0:  # Room/furniture
                draw.rectangle(
                    [px + 2, py + 2, px + scale - 3, py + scale - 3],
                    fill=color,
                    outline=None,
                )

    # Add grid
    for i in range(design.width + 1):
        x = i * scale
        draw.line([(x, 0), (x, img_height)], fill=(50, 50, 50), width=1)
    for i in range(design.height + 1):
        y = i * scale
        draw.line([(0, y), (img_width, y)], fill=(50, 50, 50), width=1)

    # Add title
    try:
        font = ImageFont.truetype("arial.ttf", 10)
        draw.text((5, 5), design.name, fill=(255, 255, 255), font=font)
    except (IOError, OSError):
        pass

    img.save(filename)


if __name__ == "__main__":
    analyze_alpha_prefabs()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw
from src.generators.prefab_analyzer import PrefabAnalyzer, PrefabDesign
import json


def test_prefab_analyzer():
    """Test the prefab analyzer with mock data"""
    
    analyzer = PrefabAnalyzer()
    
    print("Creating and analyzing mock prefabs...")
    
    # Create several mock prefabs representing different design philosophies
    prefabs = [
        create_efficient_prefab(),
        create_compact_prefab(),
        create_defensive_prefab(),
        create_luxury_prefab()
    ]
    
    # Analyze each prefab
    for prefab in prefabs:
        analyzer.analyze_prefab(prefab)
        print(f"  Analyzed: {prefab.name}")
    
    # Get learned rules
    rules = analyzer.get_learned_rules()
    
    print("\n=== LEARNED PATTERNS ===")
    
    print("\nRoom Size Statistics:")
    for room_type, stats in rules.get('room_sizes', {}).items():
        print(f"  Room Type {room_type}:")
        print(f"    Average size: {stats['mean']:.1f} tiles")
        print(f"    Range: {stats['min']}-{stats['max']} tiles")
    
    print("\nRoom Adjacency Preferences:")
    for room_type, neighbors in rules.get('adjacencies', {}).items():
        if neighbors:
            print(f"  Room Type {room_type} prefers to be next to:")
            for neighbor, probability in sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    Room Type {neighbor}: {probability:.2%}")
    
    print("\nRoom Shape Preferences:")
    for room_type, shape in rules.get('room_shapes', {}).items():
        print(f"  Room Type {room_type}: aspect ratio {shape['mean']:.2f} ± {shape['std']:.2f}")
    
    # Save patterns
    patterns_file = Path("learned_patterns.json")
    analyzer.save_learned_patterns(patterns_file)
    print(f"\nSaved learned patterns to {patterns_file}")
    
    # Generate a new layout using learned patterns
    print("\nGenerating new layout from learned patterns...")
    new_layout = analyzer.generate_from_learned_patterns(
        width=30, height=20,
        required_rooms=[3, 3, 3, 4, 5, 6, 7, 8, 9, 10]  # Bedrooms, kitchen, dining, etc.
    )
    
    # Visualize the generated layout
    visualize_layout(new_layout, "generated_from_patterns.png")
    print("Saved generated layout as generated_from_patterns.png")
    
    # Also visualize one of the analyzed prefabs for comparison
    visualize_layout(prefabs[0].layout, "analyzed_prefab_example.png")
    print("Saved example prefab as analyzed_prefab_example.png")


def create_efficient_prefab() -> PrefabDesign:
    """Create an efficient base design"""
    # Compact, well-organized layout
    layout = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,3,3,1,4,4,1,5,5,5,5,5,1,7,1],
        [1,3,3,2,4,4,2,5,5,5,5,5,2,7,1],
        [1,1,1,1,1,1,1,5,5,5,5,5,1,7,1],
        [1,3,3,1,6,6,1,1,1,2,1,1,1,7,1],
        [1,3,3,2,6,6,6,6,6,6,1,7,7,7,1],
        [1,1,1,1,6,6,6,6,6,6,2,7,7,7,1],
        [1,9,9,1,1,1,2,1,1,1,1,1,1,1,1],
        [1,9,9,9,2,8,8,8,8,1,10,10,10,1,1],
        [1,9,9,9,1,8,8,8,8,2,10,10,10,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    return PrefabDesign(
        name="efficient_base",
        width=15,
        height=11,
        layout=layout,
        metadata={"style": "efficient", "colonists": 4}
    )


def create_compact_prefab() -> PrefabDesign:
    """Create a very compact base design"""
    # Minimal space usage
    layout = np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,3,3,1,4,1,5,5,5,1],
        [1,3,3,2,4,2,5,5,5,1],
        [1,1,2,1,1,1,1,2,1,1],
        [1,6,6,6,2,7,7,7,7,1],
        [1,6,6,6,1,7,7,7,7,1],
        [1,1,1,1,1,1,1,1,1,1]
    ])
    
    return PrefabDesign(
        name="compact_base",
        width=10,
        height=7,
        layout=layout,
        metadata={"style": "compact", "colonists": 2}
    )


def create_defensive_prefab() -> PrefabDesign:
    """Create a defensive base design with thick walls"""
    # Double walls, killboxes
    layout = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1],
        [1,1,0,1,3,3,1,5,5,5,1,7,7,1,0,1,1],
        [1,1,0,1,3,3,2,5,5,5,2,7,7,1,0,1,1],
        [1,1,0,1,1,1,1,1,2,1,1,1,1,1,0,1,1],
        [1,1,0,1,6,6,6,6,6,6,6,6,6,1,0,1,1],
        [1,1,0,1,6,6,6,6,6,6,6,6,6,1,0,1,1],
        [1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1],
        [1,1,0,0,0,0,0,0,2,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    return PrefabDesign(
        name="defensive_base",
        width=17,
        height=13,
        layout=layout,
        metadata={"style": "defensive", "colonists": 3}
    )


def create_luxury_prefab() -> PrefabDesign:
    """Create a spacious luxury base design"""
    # Large rooms, lots of recreation
    layout = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,3,3,3,3,1,4,4,4,4,1,5,5,5,5,5,5,5,5,1],
        [1,3,3,3,3,2,4,4,4,4,2,5,5,5,5,5,5,5,5,1],
        [1,3,3,3,3,1,4,4,4,4,1,5,5,5,5,5,5,5,5,1],
        [1,3,3,3,3,1,4,4,4,4,1,5,5,5,5,5,5,5,5,1],
        [1,1,1,2,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,1],
        [1,8,8,8,8,8,1,6,6,6,6,6,6,1,7,7,7,7,7,1],
        [1,8,8,8,8,8,2,6,6,6,6,6,6,2,7,7,7,7,7,1],
        [1,8,8,8,8,8,1,6,6,6,6,6,6,1,7,7,7,7,7,1],
        [1,8,8,8,8,8,1,6,6,6,6,6,6,1,7,7,7,7,7,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    
    return PrefabDesign(
        name="luxury_base",
        width=20,
        height=11,
        layout=layout,
        metadata={"style": "luxury", "colonists": 6}
    )


def visualize_layout(layout: np.ndarray, filename: str):
    """Visualize a layout as an image"""
    height, width = layout.shape
    scale = 20
    
    img = Image.new('RGB', (width * scale, height * scale), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Color mapping
    colors = {
        0: (30, 30, 30),      # Empty
        1: (150, 150, 150),   # Wall
        2: (139, 69, 19),     # Door
        3: (100, 150, 200),   # Bedroom
        4: (255, 140, 0),     # Kitchen
        5: (200, 100, 50),    # Dining
        6: (255, 215, 0),     # Workshop
        7: (34, 139, 34),     # Storage
        8: (200, 100, 200),   # Recreation
        9: (255, 100, 100),   # Hospital
        10: (255, 255, 0),    # Power
    }
    
    # Draw tiles
    for y in range(height):
        for x in range(width):
            tile_type = layout[y, x]
            color = colors.get(tile_type, (100, 100, 100))
            
            px = x * scale
            py = y * scale
            
            if tile_type == 1:  # Wall
                draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                             fill=color, outline=(0, 0, 0))
            elif tile_type == 2:  # Door
                draw.rectangle([px + scale//4, py, px + 3*scale//4, py + scale - 1],
                             fill=color, outline=(0, 0, 0))
            elif tile_type > 0:  # Room
                draw.rectangle([px + 1, py + 1, px + scale - 2, py + scale - 2],
                             fill=color, outline=None)
    
    # Add grid
    for i in range(width + 1):
        x = i * scale
        draw.line([(x, 0), (x, height * scale)], fill=(50, 50, 50), width=1)
    for i in range(height + 1):
        y = i * scale
        draw.line([(0, y), (width * scale, y)], fill=(50, 50, 50), width=1)
    
    img.save(filename)


if __name__ == "__main__":
    test_prefab_analyzer()
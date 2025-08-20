"""
Test the complete RimWorld Base Assistant system.
Tests all major components working together.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.parser.save_parser import RimWorldSaveParser as SaveParser
from src.parser.mod_parser import ModParser
from src.generators.improved_wfc_generator import ImprovedWFCGenerator
from src.generators.alpha_prefab_parser import AlphaPrefabParser
from src.generators.prefab_analyzer import PrefabAnalyzer
from src.nlp.base_generator_nlp import BaseGeneratorNLP, NLPExamples
from src.generators.wfc_generator import TileType


def test_complete_system():
    """Test all components of the system"""
    print("=" * 60)
    print("RimWorld Base Assistant - Complete System Test")
    print("=" * 60)
    
    # 1. Test Save Parser with terrain
    print("\n1. Testing Save Parser...")
    save_path = Path("data/saves/Autosave-2.rws")
    if save_path.exists():
        parser = SaveParser()
        result = parser.parse(str(save_path))
        # Get first map
        if result.maps:
            first_map = result.maps[0]
            print(f"   [OK] Parsed {len(first_map.buildings)} buildings")
            colonists = first_map.get_colonists()
            print(f"   [OK] Found {len(colonists)} colonists")
            print(f"   [OK] Map size: {first_map.size}")
            if first_map.terrain:
                unique_terrains = len(set(t.def_name for t in first_map.terrain.values()))
                print(f"   [OK] Terrain grid: {unique_terrains} terrain types")
        else:
            print("   [WARNING] No maps found in save file")
    else:
        print("   [WARNING] Save file not found, skipping")
    
    # 2. Test Mod Parser
    print("\n2. Testing Mod Parser...")
    alpha_path = Path("data/AlphaPrefabs")
    if alpha_path.exists():
        mod_parser = ModParser(alpha_path)
        building_defs = mod_parser.parse_all_defs()
        print(f"   [OK] Parsed {len(building_defs)} building definitions")
        
        # Show some statistics
        beds = mod_parser.get_buildings_by_type(is_bed=True)
        tables = mod_parser.get_buildings_by_type(is_table=True)
        doors = mod_parser.get_buildings_by_type(is_door=True)
        print(f"   [OK] Found: {len(beds)} beds, {len(tables)} tables, {len(doors)} doors")
    else:
        print("   [WARNING] AlphaPrefabs mod not found, skipping")
    
    # 3. Test Prefab Analysis
    print("\n3. Testing Prefab Analysis...")
    if alpha_path.exists():
        prefab_parser = AlphaPrefabParser(alpha_path)
        layouts = prefab_parser.parse_all_layouts()
        print(f"   [OK] Found {len(layouts)} prefab layouts")
        
        # Analyze first 10
        analyzer = PrefabAnalyzer()
        for layout in layouts[:10]:
            design = prefab_parser.convert_to_prefab_design(layout)
            analyzer.analyze_prefab(design)
        
        rules = analyzer.get_learned_rules()
        print(f"   [OK] Learned patterns from {len(analyzer.patterns['room_counts'])} prefabs")
        print(f"   [OK] Room types identified: {len(rules.get('room_sizes', {}))}")
    
    # 4. Test NLP Interface
    print("\n4. Testing NLP Interface...")
    nlp = BaseGeneratorNLP(str(save_path) if save_path.exists() else None)
    
    test_inputs = [
        "Create an efficient base for 6 colonists with kitchen and storage",
        "Build a defensive fortress for 10 people with killbox",
        "Make a compact base for 4 colonists with workshop and medical bay"
    ]
    
    for test_input in test_inputs:
        requirements = nlp.parse_request(test_input)
        print(f"   [OK] Parsed: '{test_input[:50]}...'")
        print(f"     -> {requirements.num_colonists} colonists, {requirements.style} style")
    
    # 5. Test Improved WFC Generator
    print("\n5. Testing Improved WFC Generator...")
    patterns_file = Path("learned_patterns_alpha.json")
    if not patterns_file.exists():
        patterns_file = Path("learned_patterns.json")
    
    generator = ImprovedWFCGenerator(30, 30, patterns_file if patterns_file.exists() else None)
    
    # Generate a small base
    grid = generator.generate_with_templates(
        num_bedrooms=4,
        num_workrooms=2,
        include_kitchen=True,
        include_storage=True
    )
    
    # Count room types
    room_counts = {}
    for tile_value in grid.flatten():
        tile_type = TileType(tile_value)
        room_counts[tile_type.name] = room_counts.get(tile_type.name, 0) + 1
    
    print(f"   [OK] Generated {grid.shape[0]}x{grid.shape[1]} base")
    print(f"   [OK] Placed {len(generator.room_placements)} rooms")
    for room_type, count in sorted(room_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     -> {room_type}: {count} tiles")
    
    # 6. Test Complete Pipeline
    print("\n6. Testing Complete Pipeline (NLP → Generation → Visualization)...")
    
    # Generate from natural language
    user_request = "Create an efficient base for 5 colonists with kitchen, storage, and workshop"
    grid, description = nlp.generate_base(user_request, width=40, height=40)
    
    print(f"   [OK] Generated base from: '{user_request}'")
    print("   [OK] Description:")
    for line in description.split('\n')[:5]:
        print(f"     {line}")
    
    # Save visualization
    visualize_grid(grid, "test_complete_system.png")
    print(f"   [OK] Saved visualization to test_complete_system.png")
    
    print("\n" + "=" * 60)
    print("All system components tested successfully!")
    print("=" * 60)


def visualize_grid(grid: np.ndarray, filename: str):
    """Create a simple visualization of the generated grid"""
    scale = 10
    height, width = grid.shape
    img = Image.new('RGB', (width * scale, height * scale), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Color mapping
    colors = {
        TileType.EMPTY: (30, 30, 30),
        TileType.WALL: (150, 150, 150),
        TileType.DOOR: (139, 69, 19),
        TileType.CORRIDOR: (100, 100, 100),
        TileType.BEDROOM: (100, 150, 200),
        TileType.KITCHEN: (255, 140, 0),
        TileType.STORAGE: (0, 200, 0),
        TileType.WORKSHOP: (255, 215, 0),
        TileType.RECREATION: (218, 112, 214),
        TileType.MEDICAL: (255, 107, 107),
        TileType.RESEARCH: (147, 112, 219),
        TileType.POWER: (255, 255, 0)
    }
    
    for y in range(height):
        for x in range(width):
            tile_type = TileType(grid[y, x])
            color = colors.get(tile_type, (100, 100, 100))
            
            px = x * scale
            py = y * scale
            
            draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                         fill=color, outline=(0, 0, 0))
    
    img.save(filename)


if __name__ == "__main__":
    test_complete_system()
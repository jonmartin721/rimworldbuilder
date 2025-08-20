import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree

def check_terrain_structure():
    tree = etree.parse('data/saves/Autosave-2.rws')
    root = tree.getroot()
    game = root.find('game')
    maps = game.find('maps')
    first_map = maps.find('li')
    
    # Check terrainGrid structure
    terrain_grid = first_map.find('terrainGrid')
    if terrain_grid is not None:
        print("terrainGrid structure:")
        for child in terrain_grid:
            print(f"  {child.tag}: {child.text[:50] if child.text else 'No text'}...")
            if child.tag == 'tiles' and child.text:
                # Check if it's a list of terrain defs
                lines = child.text.strip().split('\n')
                print(f"    Total lines: {len(lines)}")
                # Check first few and look for bridges
                bridge_count = 0
                for i, line in enumerate(lines):
                    if 'Bridge' in line:
                        bridge_count += 1
                        if bridge_count <= 3:
                            print(f"    Line {i}: {line}")
                print(f"    Total bridge tiles: {bridge_count}")
    
    # Check underGrid structure
    under_grid = first_map.find('underGrid')
    if under_grid is not None:
        print("\nunderGrid structure:")
        for child in under_grid:
            print(f"  {child.tag}: {child.text[:50] if child.text else 'No text'}...")
            if child.tag == 'tiles' and child.text:
                lines = child.text.strip().split('\n')
                print(f"    Total lines: {len(lines)}")
                bridge_count = 0
                for line in lines:
                    if 'Bridge' in line:
                        bridge_count += 1
                print(f"    Total bridge under-tiles: {bridge_count}")

check_terrain_structure()
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.terrain_decoder import TerrainDecoder
from lxml import etree
import numpy as np

# Load save file
tree = etree.parse('data/saves/Autosave-2.rws')
root = tree.getroot()
game = root.find('game')
maps = game.find('maps')
first_map = maps.find('li')

decoder = TerrainDecoder()
terrain_grid = decoder.decode_terrain_grid(first_map)
under_grid = decoder.decode_under_grid(first_map)

if terrain_grid is not None:
    # Get unique terrain IDs
    unique_ids = np.unique(terrain_grid)
    
    print("Analyzing terrain patterns to identify bridges...")
    print("\nTerrain IDs by occurrence count:")
    
    terrain_counts = []
    for tid in unique_ids:
        count = np.sum(terrain_grid == tid)
        terrain_counts.append((tid, count))
    
    # Sort by count
    terrain_counts.sort(key=lambda x: x[1])
    
    for tid, count in terrain_counts:
        hex_id = f"0x{tid:04X}"
        known_name = decoder.TERRAIN_IDS.get(tid, "")
        
        # Look for patterns that suggest bridges
        # Bridges are typically:
        # - Not super common (not base terrain)
        # - Not super rare (not single tiles)
        # - Often in the 50-500 tile range
        
        if 50 <= count <= 500:
            print(f"  {hex_id} ({tid:5}): {count:5} tiles - POSSIBLE BRIDGE? {known_name}")
        else:
            print(f"  {hex_id} ({tid:5}): {count:5} tiles - {known_name}")
    
    # Check the locations of medium-occurrence terrains
    print("\n\nChecking spatial patterns of possible bridges...")
    
    for tid, count in terrain_counts:
        if 50 <= count <= 500:
            # Get positions
            positions = np.argwhere(terrain_grid == tid)
            
            # Check if they form lines (bridges are often linear)
            if len(positions) > 10:
                # Calculate variance in x and y
                x_coords = positions[:, 1]
                y_coords = positions[:, 0]
                
                x_var = np.var(x_coords)
                y_var = np.var(y_coords)
                
                # Check if mostly horizontal or vertical
                if x_var < 100 or y_var < 100:
                    print(f"\n  ID 0x{tid:04X} forms a linear pattern!")
                    print(f"    X variance: {x_var:.1f}, Y variance: {y_var:.1f}")
                    print(f"    Sample positions: {positions[:5].tolist()}")
                    
                    # This is likely a bridge or road!
    
    # Check if we know about Frame_HeavyBridge from the things
    print("\n\nCross-referencing with known bridge positions from things...")
    
    # We know Frame_HeavyBridge frames are at certain positions
    # Let's see what terrain is under them
    
    things = first_map.find('things')
    if things is not None:
        bridge_positions = []
        for thing in things.findall('thing'):
            def_elem = thing.find('def')
            if def_elem is not None and 'Bridge' in str(def_elem.text):
                pos = thing.find('pos')
                if pos is not None and pos.text:
                    # Parse position (x, 0, z) format
                    coords = pos.text.strip('()').split(',')
                    if len(coords) == 3:
                        x = int(float(coords[0]))
                        z = int(float(coords[2]))  # z is y in our grid
                        bridge_positions.append((x, z))
        
        if bridge_positions:
            print(f"Found {len(bridge_positions)} bridge things")
            
            # Check what terrain IDs are at those positions
            terrain_at_bridges = set()
            for x, y in bridge_positions[:10]:  # Check first 10
                if 0 <= x < 275 and 0 <= y < 275:
                    tid = terrain_grid[y, x]
                    terrain_at_bridges.add(tid)
                    print(f"  Bridge at ({x},{y}) has terrain ID 0x{tid:04X}")
            
            if terrain_at_bridges:
                print(f"\nBridge positions have these terrain IDs: {[f'0x{t:04X}' for t in terrain_at_bridges]}")
                print("These are likely the completed bridge terrain IDs!")
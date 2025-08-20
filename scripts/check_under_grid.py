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

if under_grid is not None:
    print("Analyzing under grid (terrain under buildings)...")
    
    # Get unique IDs in under grid
    unique_under = np.unique(under_grid)
    print(f"\nUnique under-terrain IDs: {len(unique_under)}")
    
    for tid in unique_under:
        count = np.sum(under_grid == tid)
        print(f"  0x{tid:04X} ({tid}): {count} tiles")
    
    # Check bridge positions in under grid
    print("\n\nChecking under-grid at known bridge positions...")
    
    things = first_map.find('things')
    if things is not None:
        bridge_positions = []
        for thing in things.findall('thing'):
            def_elem = thing.find('def')
            if def_elem is not None and def_elem.text == 'Frame_HeavyBridge':
                pos = thing.find('pos')
                if pos is not None and pos.text:
                    coords = pos.text.strip('()').split(',')
                    if len(coords) == 3:
                        x = int(float(coords[0]))
                        z = int(float(coords[2]))
                        bridge_positions.append((x, z))
        
        if bridge_positions:
            print(f"Checking {len(bridge_positions)} Frame_HeavyBridge positions...")
            
            # Check under grid at those positions
            under_at_bridges = {}
            for x, y in bridge_positions:
                if 0 <= x < 275 and 0 <= y < 275:
                    tid = under_grid[y, x]
                    under_at_bridges[tid] = under_at_bridges.get(tid, 0) + 1
            
            print("\nUnder-terrain at Frame_HeavyBridge positions:")
            for tid, count in sorted(under_at_bridges.items(), key=lambda x: x[1], reverse=True):
                print(f"  0x{tid:04X}: {count} positions")
                
                # If this is not 0xFF89, it might be the completed bridge ID
                if tid != 0xFF89:
                    print(f"    ^ This could be completed HeavyBridge terrain!")
    
    # Let's also check for HeavyBridge (non-Frame) in things
    print("\n\nLooking for completed HeavyBridge things...")
    
    if things is not None:
        completed_bridges = []
        for thing in things.findall('thing'):
            def_elem = thing.find('def')
            if def_elem is not None and def_elem.text == 'HeavyBridge':
                pos = thing.find('pos')
                if pos is not None and pos.text:
                    coords = pos.text.strip('()').split(',')
                    if len(coords) == 3:
                        x = int(float(coords[0]))
                        z = int(float(coords[2]))
                        completed_bridges.append((x, z))
        
        if completed_bridges:
            print(f"Found {len(completed_bridges)} completed HeavyBridge things!")
            
            # Check what terrain they have
            for x, y in completed_bridges[:5]:
                if 0 <= x < 275 and 0 <= y < 275:
                    top_tid = terrain_grid[y, x]
                    under_tid = under_grid[y, x]
                    print(f"  At ({x},{y}): top=0x{top_tid:04X}, under=0x{under_tid:04X}")
        else:
            print("No completed HeavyBridge things found (they might be in terrain only)")
    
    # Final check: positions where under != top might be interesting
    print("\n\nPositions where under-grid differs from top-grid (sample):")
    diff_positions = np.argwhere(under_grid != terrain_grid)
    
    if len(diff_positions) > 0:
        print(f"Found {len(diff_positions)} positions with different under/top terrain")
        
        # Sample a few
        for i in range(min(10, len(diff_positions))):
            y, x = diff_positions[i]
            top_tid = terrain_grid[y, x]
            under_tid = under_grid[y, x]
            print(f"  ({x},{y}): top=0x{top_tid:04X}, under=0x{under_tid:04X}")
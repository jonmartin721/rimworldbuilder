import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib
import struct
import numpy as np
from src.parser.terrain_decoder import TerrainDecoder

def decode_foundation():
    tree = etree.parse('data/saves/Autosave-2.rws')
    root = tree.getroot()
    game = root.find('game')
    maps = game.find('maps')
    first_map = maps.find('li')
    
    # Decode terrain for reference
    decoder = TerrainDecoder()
    terrain_grid = decoder.decode_terrain_grid(first_map)
    
    # Decode foundation grid
    terrain_elem = first_map.find('terrainGrid')
    if terrain_elem is not None:
        foundation_elem = terrain_elem.find('foundationGridDeflate')
        if foundation_elem is not None and foundation_elem.text:
            compressed = base64.b64decode(foundation_elem.text.strip())
            decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
            
            # Parse as 16-bit values
            num_tiles = len(decompressed) // 2
            foundation_values = struct.unpack(f'<{num_tiles}H', decompressed)
            
            # Convert to 2D array
            foundation_grid = np.array(foundation_values, dtype=np.uint16)
            foundation_grid = foundation_grid.reshape((275, 275))
            
            print("Foundation Grid Analysis:")
            print("========================")
            
            # Count each value
            unique, counts = np.unique(foundation_grid, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"  0x{val:04X} ({val}): {count} tiles")
            
            # Find positions of non-zero foundation
            print("\n\nPositions with foundations (bridges):")
            print("=====================================")
            
            # Check 0x1A47 positions
            positions_1a47 = np.argwhere(foundation_grid == 0x1A47)
            print(f"\n0x1A47 foundation: {len(positions_1a47)} positions")
            if len(positions_1a47) > 0:
                print(f"  First few positions: {positions_1a47[:5].tolist()}")
                
                # Check what terrain is under these
                terrain_under_1a47 = set()
                for y, x in positions_1a47:
                    tid = terrain_grid[y, x]
                    terrain_under_1a47.add(tid)
                print(f"  Terrain IDs under 0x1A47: {[f'0x{t:04X}' for t in sorted(terrain_under_1a47)]}")
                
                # Check if these form lines (bridges usually do)
                x_coords = positions_1a47[:, 1]
                y_coords = positions_1a47[:, 0]
                x_var = np.var(x_coords)
                y_var = np.var(y_coords)
                print(f"  Spatial pattern - X variance: {x_var:.1f}, Y variance: {y_var:.1f}")
                
                if x_var < 500 or y_var < 500:
                    print("  -> Forms linear/clustered pattern (likely bridges!)")
            
            # Check 0x8C7D positions
            positions_8c7d = np.argwhere(foundation_grid == 0x8C7D)
            print(f"\n0x8C7D foundation: {len(positions_8c7d)} positions")
            if len(positions_8c7d) > 0:
                print(f"  First few positions: {positions_8c7d[:5].tolist()}")
                
                # Check what terrain is under these
                terrain_under_8c7d = set()
                for y, x in positions_8c7d:
                    tid = terrain_grid[y, x]
                    terrain_under_8c7d.add(tid)
                print(f"  Terrain IDs under 0x8C7D: {[f'0x{t:04X}' for t in sorted(terrain_under_8c7d)]}")
                
                # Check pattern
                x_coords = positions_8c7d[:, 1]
                y_coords = positions_8c7d[:, 0]
                x_var = np.var(x_coords)
                y_var = np.var(y_coords)
                print(f"  Spatial pattern - X variance: {x_var:.1f}, Y variance: {y_var:.1f}")
                
                if x_var < 500 or y_var < 500:
                    print("  -> Forms linear/clustered pattern (likely bridges!)")
            
            # Cross-reference with known Frame_HeavyBridge positions
            print("\n\nCross-reference with Frame_HeavyBridge positions:")
            print("=================================================")
            
            things = first_map.find('things')
            if things is not None:
                frame_positions = []
                for thing in things.findall('thing'):
                    def_elem = thing.find('def')
                    if def_elem is not None and def_elem.text == 'Frame_HeavyBridge':
                        pos = thing.find('pos')
                        if pos is not None and pos.text:
                            coords = pos.text.strip('()').split(',')
                            if len(coords) == 3:
                                x = int(float(coords[0]))
                                z = int(float(coords[2]))
                                frame_positions.append((x, z))
                
                if frame_positions:
                    print(f"Checking {len(frame_positions)} Frame_HeavyBridge positions...")
                    
                    # Check foundation at these positions
                    foundation_at_frames = {}
                    for x, y in frame_positions:
                        if 0 <= x < 275 and 0 <= y < 275:
                            fval = foundation_grid[y, x]
                            foundation_at_frames[fval] = foundation_at_frames.get(fval, 0) + 1
                    
                    print("Foundation values at Frame_HeavyBridge positions:")
                    for fval, count in foundation_at_frames.items():
                        print(f"  0x{fval:04X}: {count} positions")
            
            # Show sample of bridge locations
            print("\n\nSample bridge locations on map:")
            print("================================")
            
            # Show a few positions of each type
            if len(positions_1a47) > 0:
                print("\n0x1A47 bridges (Heavy/Stone?):")
                for i in range(min(5, len(positions_1a47))):
                    y, x = positions_1a47[i]
                    tid = terrain_grid[y, x]
                    terrain_name = decoder.get_terrain_name(tid)
                    print(f"  Position ({x:3}, {y:3}): terrain=0x{tid:04X} {terrain_name}")
            
            if len(positions_8c7d) > 0:
                print("\n0x8C7D bridges (Wood/Light?):")
                for i in range(min(5, len(positions_8c7d))):
                    y, x = positions_8c7d[i]
                    tid = terrain_grid[y, x]
                    terrain_name = decoder.get_terrain_name(tid)
                    print(f"  Position ({x:3}, {y:3}): terrain=0x{tid:04X} {terrain_name}")
            
            return foundation_grid

foundation = decode_foundation()
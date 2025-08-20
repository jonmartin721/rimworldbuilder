import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib

def decode_terrain():
    tree = etree.parse('data/saves/Autosave-2.rws')
    root = tree.getroot()
    game = root.find('game')
    maps = game.find('maps')
    first_map = maps.find('li')
    
    # Get terrain definitions first
    defs = root.find('.//defs')
    terrain_defs = []
    if defs is not None:
        for terrain_def in defs.findall('.//TerrainDef'):
            def_name = terrain_def.find('defName')
            if def_name is not None:
                terrain_defs.append(def_name.text)
    
    print(f"Found {len(terrain_defs)} terrain definitions in save")
    
    map_width = 250
    map_height = 250
    
    terrain_grid = first_map.find('terrainGrid')
    if terrain_grid is not None:
        top_grid = terrain_grid.find('topGridDeflate')
        if top_grid is not None and top_grid.text:
            data = top_grid.text.strip()
            
            # Decode and decompress
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)
            
            print(f"Decompressed size: {len(decompressed)} bytes")
            
            # The pattern 90a7 appears to be repeating - this is likely a serialized format
            # RimWorld uses a custom binary format for terrain
            
            # Let's look for strings in the data
            strings = []
            current = b''
            for i, byte in enumerate(decompressed):
                if 32 <= byte < 127:  # Printable ASCII
                    current += bytes([byte])
                else:
                    if len(current) > 3:  # Only keep strings longer than 3 chars
                        try:
                            s = current.decode('ascii')
                            strings.append((i - len(current), s))
                        except:
                            pass
                    current = b''
            
            print(f"\nFound {len(strings)} strings in data")
            
            # Filter for terrain-like strings
            terrain_strings = []
            for pos, s in strings:
                if any(terrain_word in s for terrain_word in ['Soil', 'Sand', 'Gravel', 'Marsh', 'Water', 'Bridge', 'Floor', 'Stone', 'Concrete']):
                    terrain_strings.append((pos, s))
                    if 'Bridge' in s:
                        print(f"  Bridge string at byte {pos}: '{s}'")
            
            print(f"\nFound {len(terrain_strings)} terrain-related strings")
            
            # RimWorld serialization format analysis
            # The data starts with repeating 0x90 0xa7 pattern
            # This might be a dictionary/lookup table format
            
            # Let's find where the actual tile data starts
            # Usually after the terrain type definitions
            
            # Look for the end of repeated pattern
            pattern_end = 0
            for i in range(0, min(1000, len(decompressed)), 2):
                if decompressed[i:i+2] != b'\x90\xa7':
                    pattern_end = i
                    break
            
            print(f"\nRepeating pattern ends at byte {pattern_end}")
            
            # The actual format appears to be:
            # 1. Header/metadata (possibly map size)
            # 2. Terrain type dictionary
            # 3. Tile indices or references
            
            # Let's search more specifically for bridge data
            # Try to find where terrain assignments start
            
            # Method 1: Look for terrain def names as a table
            print("\n--- Searching for terrain definition table ---")
            
            # Common RimWorld terrain types
            terrain_types = [
                b'Soil', b'SoilRich', b'Sand', b'Gravel', b'MarshyTerrain',
                b'WaterShallow', b'WaterDeep', b'WaterOceanShallow',
                b'HeavyBridge', b'Bridge', b'SteelBridge',
                b'Concrete', b'PavedTile', b'WoodPlankFloor',
                b'TileStone', b'Flagstone'
            ]
            
            found_terrains = {}
            for terrain in terrain_types:
                if terrain in decompressed:
                    count = decompressed.count(terrain)
                    idx = decompressed.find(terrain)
                    found_terrains[terrain.decode()] = (count, idx)
                    print(f"  {terrain.decode()}: {count} occurrences, first at byte {idx}")
            
            # Focus on HeavyBridge
            if b'HeavyBridge' in decompressed:
                print("\n=== HEAVYBRIDGE ANALYSIS ===")
                
                # Find all occurrences
                bridge_positions = []
                idx = 0
                while True:
                    idx = decompressed.find(b'HeavyBridge', idx)
                    if idx == -1:
                        break
                    
                    # Get context
                    before = decompressed[max(0, idx-10):idx]
                    after = decompressed[idx:idx+20]
                    
                    bridge_positions.append({
                        'pos': idx,
                        'before': before.hex(),
                        'after': after.hex(),
                        'text': after.decode('utf-8', errors='replace')
                    })
                    
                    idx += 1
                
                print(f"Found {len(bridge_positions)} HeavyBridge occurrences")
                
                for i, bp in enumerate(bridge_positions[:3]):
                    print(f"\nOccurrence {i+1} at byte {bp['pos']}:")
                    print(f"  Before (hex): {bp['before']}")
                    print(f"  After (hex): {bp['after']}")
                    print(f"  Text: {bp['text']}")
                
                return bridge_positions

result = decode_terrain()
if result:
    print(f"\n=== SUCCESS: Found {len(result)} bridge references ===")
    print("Bridges are in the terrain data but in a complex serialized format")
    print("Would need full RimWorld save format documentation to extract positions")
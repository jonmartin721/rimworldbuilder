import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree

tree = etree.parse('data/saves/Autosave-2.rws')
root = tree.getroot()
game = root.find('game')
maps = game.find('maps')
first_map = maps.find('li')

# Find actual map size
map_info = first_map.find('mapInfo')
if map_info is not None:
    size = map_info.find('size')
    if size is not None:
        x = size.find('x')
        z = size.find('z')
        if x is not None and z is not None:
            print(f"Map size from mapInfo: {x.text} x {z.text}")
            print(f"Total tiles: {int(x.text) * int(z.text)}")
            print(f"Expected bytes (2 per tile): {int(x.text) * int(z.text) * 2}")
        
# Check actual decompressed size
import base64
import zlib

terrain_grid = first_map.find('terrainGrid')
if terrain_grid is not None:
    top_grid = terrain_grid.find('topGridDeflate')
    if top_grid is not None and top_grid.text:
        compressed = base64.b64decode(top_grid.text.strip())
        decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
        print(f"\nActual decompressed size: {len(decompressed)} bytes")
        print(f"This equals: {len(decompressed) // 2} tiles")
        
        # Calculate possible map dimensions
        num_tiles = len(decompressed) // 2
        print(f"\nPossible square map size: {int(num_tiles ** 0.5)}")
        
        # Check if it's 275x275
        if num_tiles == 275 * 275:
            print("Map is 275x275!")
        elif num_tiles == 250 * 250:
            print("Map is 250x250!")
        else:
            # Try to factor it
            for w in [250, 275, 300, 200, 225]:
                if num_tiles % w == 0:
                    h = num_tiles // w
                    print(f"Could be {w}x{h}")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib


def parse_terrain():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    # Get map size
    map_info = first_map.find("mapInfo")
    map_width = 250
    map_height = 250
    if map_info is not None:
        size = map_info.find("size")
        if size is not None:
            x_elem = size.find("x")
            z_elem = size.find("z")
            if x_elem is not None and z_elem is not None:
                map_width = int(x_elem.text)
                map_height = int(z_elem.text)

    print(f"Map size: {map_width}x{map_height}")
    expected_tiles = map_width * map_height
    print(f"Expected tiles: {expected_tiles}")

    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        top_grid = terrain_grid.find("topGridDeflate")
        if top_grid is not None and top_grid.text:
            data = top_grid.text.strip()

            # Decode and decompress
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)

            print(f"\nDecompressed size: {len(decompressed)} bytes")

            # The data appears to be a sequence of terrain def names
            # Let's try different parsing approaches

            # 1. Try as null-terminated strings
            terrain_tiles = []
            current = b""
            for byte in decompressed:
                if byte == 0:
                    if current:
                        terrain_tiles.append(current.decode("utf-8", errors="ignore"))
                    current = b""
                else:
                    current += bytes([byte])

            if current:
                terrain_tiles.append(current.decode("utf-8", errors="ignore"))

            print(f"\nParsed as null-terminated strings: {len(terrain_tiles)} tiles")

            if len(terrain_tiles) > 0:
                # Show unique terrain types
                unique_terrains = set(terrain_tiles)
                print(f"Unique terrain types: {len(unique_terrains)}")

                # Show first few
                print("\nFirst 10 terrain types:")
                for terrain in list(unique_terrains)[:10]:
                    count = terrain_tiles.count(terrain)
                    print(f"  {terrain}: {count} tiles")

                # Find bridges
                bridge_tiles = []
                for i, terrain in enumerate(terrain_tiles):
                    if "Bridge" in terrain:
                        x = i % map_width
                        y = i // map_width
                        bridge_tiles.append((x, y, terrain))

                print(f"\n=== FOUND {len(bridge_tiles)} BRIDGE TILES ===")
                if bridge_tiles:
                    print("First 10 bridge positions:")
                    for x, y, terrain in bridge_tiles[:10]:
                        print(f"  ({x}, {y}): {terrain}")

                    # Group by type
                    bridge_types = {}
                    for x, y, terrain in bridge_tiles:
                        bridge_types[terrain] = bridge_types.get(terrain, 0) + 1

                    print("\nBridge types found:")
                    for bridge_type, count in bridge_types.items():
                        print(f"  {bridge_type}: {count} tiles")

                    return bridge_tiles

            # 2. If that didn't work, try as UTF-16
            if len(terrain_tiles) != expected_tiles:
                print("\nTrying UTF-16 decode...")
                try:
                    text = decompressed.decode("utf-16", errors="ignore")
                    print(f"UTF-16 text length: {len(text)}")
                except UnicodeDecodeError:
                    print("UTF-16 decode failed")

            # 3. Try as fixed-length records
            if len(terrain_tiles) != expected_tiles:
                print("\nTrying as fixed-length records...")
                # Each tile might be stored as a fixed number of bytes
                bytes_per_tile = len(decompressed) // expected_tiles
                print(
                    f"Bytes per tile: {bytes_per_tile} (with {len(decompressed) % expected_tiles} remainder)"
                )


parse_terrain()

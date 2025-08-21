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
    map_width = 250
    map_height = 250

    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        top_grid = terrain_grid.find("topGridDeflate")
        if top_grid is not None and top_grid.text:
            data = top_grid.text.strip()

            # Decode and decompress
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)

            print(f"Decompressed size: {len(decompressed)} bytes")
            print(f"Expected tiles: {map_width * map_height}")

            # Look for the pattern - RimWorld uses a specific format
            # Let's examine the data structure
            print(f"\nFirst 100 bytes (hex): {decompressed[:100].hex()}")

            # Check for HeavyBridge occurrences
            if b"HeavyBridge" in decompressed:
                count = decompressed.count(b"HeavyBridge")
                print(f"\nFound {count} occurrences of 'HeavyBridge' in terrain data!")

                # Find all positions
                positions = []
                idx = 0
                while True:
                    idx = decompressed.find(b"HeavyBridge", idx)
                    if idx == -1:
                        break
                    positions.append(idx)
                    idx += 1

                print(f"Byte positions of 'HeavyBridge': {positions[:10]}...")

                # The data seems to be a serialized format
                # Let's look at the pattern around HeavyBridge
                if positions:
                    first_pos = positions[0]
                    context = decompressed[max(0, first_pos - 20) : first_pos + 30]
                    print("\nContext around first HeavyBridge:")
                    print(f"  Raw bytes: {context}")
                    print(f"  As text: {context.decode('utf-8', errors='replace')}")

            # Try another approach - RimWorld might use length-prefixed strings
            print("\n--- Trying length-prefixed string format ---")

            offset = 0
            terrain_list = []
            tile_count = 0

            while offset < len(decompressed) and tile_count < map_width * map_height:
                # Try reading as: [1 byte length][string data]
                if offset >= len(decompressed):
                    break

                str_len = decompressed[offset]
                offset += 1

                if str_len == 0:
                    # Null terrain
                    terrain_list.append("")
                elif offset + str_len <= len(decompressed):
                    terrain_data = decompressed[offset : offset + str_len]
                    try:
                        terrain = terrain_data.decode("utf-8", errors="ignore")
                        terrain_list.append(terrain)
                        if "Bridge" in terrain:
                            x = tile_count % map_width
                            y = tile_count // map_width
                            print(f"  Found bridge at ({x}, {y}): {terrain}")
                    except (IndexError, ValueError):
                        terrain_list.append("")
                    offset += str_len
                else:
                    break

                tile_count += 1

            print(f"\nParsed {tile_count} tiles using length-prefixed format")

            # Try yet another format - look for a repeated pattern
            print("\n--- Analyzing data structure ---")

            # Check if it's actually a different format altogether
            # Some games use run-length encoding or indexed terrain

            # Look for repeating patterns
            if len(decompressed) > 1000:
                # Check if there's a header

                # Common header sizes
                for header_size in [0, 4, 8, 12, 16, 20]:
                    data_start = header_size
                    data_section = decompressed[data_start:]

                    # Check if data section divides evenly
                    if len(data_section) % (map_width * map_height) == 0:
                        bytes_per_tile = len(data_section) // (map_width * map_height)
                        print(
                            f"\nWith {header_size} byte header: {bytes_per_tile} bytes per tile"
                        )

                        if bytes_per_tile in [1, 2, 4]:
                            # This might be an index format
                            print(
                                f"  Might be {bytes_per_tile}-byte indices into a terrain table"
                            )


parse_terrain()

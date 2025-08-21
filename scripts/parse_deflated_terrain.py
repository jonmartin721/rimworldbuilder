import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib


def parse_deflated_terrain():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    # Get map size
    map_info = first_map.find("mapInfo")
    if map_info:
        size = map_info.find("size")
        if size:
            x = int(size.find("x").text)
            z = int(size.find("z").text)
            print(f"Map size: {x}x{z}")
            x * z

    # Parse topGridDeflate (main terrain)
    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        top_grid = terrain_grid.find("topGridDeflate")
        if top_grid is not None and top_grid.text:
            try:
                # Decode from custom base64
                compressed = base64.b64decode(top_grid.text)
                decompressed = zlib.decompress(compressed)

                # Parse as null-separated strings
                terrain_defs = decompressed.decode("utf-8", errors="ignore").split(
                    "\x00"
                )

                print(f"\nTerrain tiles: {len(terrain_defs)} entries")

                # Count bridge tiles
                bridge_positions = []
                for i, terrain in enumerate(terrain_defs):
                    if "Bridge" in terrain:
                        x = i % 250  # Assuming 250x250 map
                        y = i // 250
                        bridge_positions.append((x, y, terrain))

                print(f"Found {len(bridge_positions)} bridge terrain tiles")
                if bridge_positions:
                    print("First few bridge positions:")
                    for x, y, terrain in bridge_positions[:5]:
                        print(f"  ({x}, {y}): {terrain}")

            except Exception as e:
                print(f"Error decoding topGridDeflate: {e}")

    # Parse underGridDeflate (terrain under buildings)
    if terrain_grid is not None:
        under_grid = terrain_grid.find("underGridDeflate")
        if under_grid is not None and under_grid.text:
            try:
                compressed = base64.b64decode(under_grid.text)
                decompressed = zlib.decompress(compressed)

                under_defs = decompressed.decode("utf-8", errors="ignore").split("\x00")

                print(f"\nUnder-grid tiles: {len(under_defs)} entries")

                bridge_count = sum(1 for terrain in under_defs if "Bridge" in terrain)
                print(f"Found {bridge_count} bridge under-tiles")

            except Exception as e:
                print(f"Error decoding underGridDeflate: {e}")


parse_deflated_terrain()

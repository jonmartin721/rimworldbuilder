import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib


def parse_terrain_for_bridges():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    # Check terrainGrid
    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        # Check for compressed data
        data_elem = terrain_grid.find("data")
        if data_elem is not None and data_elem.text:
            try:
                # Try to decode as base64 + zlib
                compressed = base64.b64decode(data_elem.text)
                decompressed = zlib.decompress(compressed)

                # Look for bridge in decompressed data
                if b"HeavyBridge" in decompressed:
                    print("Found HeavyBridge in terrain data!")
                    # Count occurrences
                    count = decompressed.count(b"HeavyBridge")
                    print(f"  Found {count} HeavyBridge terrain tiles")

                    # Show some context
                    idx = decompressed.find(b"HeavyBridge")
                    sample = decompressed[max(0, idx - 50) : idx + 50]
                    print(f"  Sample: {sample}")

            except Exception as e:
                print(f"Could not decompress terrain data: {e}")
                # Try as raw text
                if "HeavyBridge" in data_elem.text:
                    count = data_elem.text.count("HeavyBridge")
                    print(f"Found {count} HeavyBridge references in raw terrain data")

    # Check underGrid (buildable terrain under structures)
    under_grid = first_map.find("underGrid")
    if under_grid is not None:
        data_elem = under_grid.find("data")
        if data_elem is not None and data_elem.text:
            try:
                compressed = base64.b64decode(data_elem.text)
                decompressed = zlib.decompress(compressed)

                if b"HeavyBridge" in decompressed:
                    print("\nFound HeavyBridge in underGrid data!")
                    count = decompressed.count(b"HeavyBridge")
                    print(f"  Found {count} HeavyBridge under-tiles")

            except Exception as e:
                print(f"Could not decompress underGrid: {e}")


parse_terrain_for_bridges()

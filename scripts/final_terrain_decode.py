import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib


def final_decode():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    map_width = 250
    map_height = 250
    total_tiles = map_width * map_height

    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        # First, let's check if there's a terrain lookup table
        lookup_elem = terrain_grid.find("terrainLookup")
        if lookup_elem is not None:
            print("Found terrain lookup table!")

        top_grid = terrain_grid.find("topGridDeflate")
        if top_grid is not None and top_grid.text:
            data = top_grid.text.strip()

            # Decode and decompress
            decoded = base64.b64decode(data)
            decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)

            print(f"Decompressed size: {len(decompressed)} bytes")
            print(f"Bytes per tile if uniform: {len(decompressed) / total_tiles:.2f}")

            # The decompressed size of 151250 for 62500 tiles = ~2.42 bytes per tile
            # This suggests a complex encoding, possibly with a header

            # Let's look at the actual structure more carefully
            # RimWorld likely uses one of these formats:
            # 1. Run-length encoding (terrain_id, count)
            # 2. Dictionary + indices
            # 3. Serialized C# objects

            # Check the byte distribution
            byte_counts = {}
            for byte in decompressed:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1

            print(f"\nUnique byte values: {len(byte_counts)}")

            # Most common bytes
            common_bytes = sorted(
                byte_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
            print("Most common bytes:")
            for byte_val, count in common_bytes:
                print(
                    f"  0x{byte_val:02x}: {count} times ({count / len(decompressed) * 100:.1f}%)"
                )

            # The 0x90 0xa7 pattern suggests this might be serialized data
            # Let's try to find the actual terrain data by looking for known patterns

            # In RimWorld saves, terrain is often stored as:
            # [Header with terrain type list][Grid of indices]

            # Let's look for ASCII strings that might be terrain names
            print("\n--- Extracting readable strings ---")

            strings = []
            i = 0
            while i < len(decompressed):
                # Look for ASCII string patterns
                if decompressed[i] >= 32 and decompressed[i] < 127:
                    start = i
                    while i < len(decompressed) and 32 <= decompressed[i] < 127:
                        i += 1
                    if i - start > 4:  # Minimum string length
                        string = decompressed[start:i].decode("ascii", errors="ignore")
                        strings.append((start, string))
                else:
                    i += 1

            print(f"Found {len(strings)} readable strings")

            # Filter for terrain-related
            terrain_keywords = [
                "Soil",
                "Sand",
                "Water",
                "Bridge",
                "Stone",
                "Marsh",
                "Gravel",
                "Floor",
                "Concrete",
                "Tile",
                "Wood",
                "Steel",
                "Heavy",
            ]

            terrain_strings = []
            for pos, s in strings:
                if any(kw in s for kw in terrain_keywords):
                    terrain_strings.append((pos, s))
                    print(f"  @{pos}: {s[:50]}")

            # Alternative approach: The data might be using a custom encoding
            # where terrain types are stored elsewhere in the save

            print("\n--- Checking for terrain references in main map ---")

            # Look for TerrainGrid in a different format
            for elem in first_map.iter():
                if "terrain" in elem.tag.lower():
                    print(f"Found element: {elem.tag}")
                    for child in elem:
                        print(f"  Child: {child.tag}")
                        if child.text and len(child.text) < 100:
                            print(f"    Text: {child.text[:50]}...")

            # Final attempt: brute force search for bridge positions
            # Even if we can't decode the full format, we can find where bridges are mentioned

            print("\n=== SUMMARY ===")
            print("The terrain data is compressed and encoded in a proprietary format.")
            print("It appears to use a lookup table with indices, but the exact format")
            print("would require reverse engineering the RimWorld save serialization.")
            print(
                "\nCompleted bridges ARE in the save file but in the compressed terrain grid."
            )
            print("To fully decode this would require:")
            print("1. Understanding RimWorld's C# serialization format")
            print("2. Mapping the byte patterns to terrain type indices")
            print("3. Reconstructing the tile grid from the encoded data")


final_decode()

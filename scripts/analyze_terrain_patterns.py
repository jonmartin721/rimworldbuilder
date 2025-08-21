import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gzip
import base64
import zlib
from collections import Counter
from lxml import etree


def analyze_terrain_patterns(save_path: Path):
    """Analyze terrain byte patterns to identify terrain types"""
    print(f"Analyzing terrain patterns in: {save_path}")

    if not save_path.exists():
        print(f"Save file not found: {save_path}")
        return

    # Open and parse the save file
    with open(save_path, "rb") as f:
        magic_bytes = f.read(2)
        f.seek(0)

        if magic_bytes == b"\x1f\x8b":
            content = gzip.decompress(f.read())
        else:
            content = f.read()

    # Parse XML
    root = etree.fromstring(content)

    # Find terrain grid
    terrain_grid = root.find(".//terrainGrid")
    if terrain_grid is None:
        print("No terrainGrid found")
        return

    # Get the deflated terrain data
    top_grid = terrain_grid.find("topGridDeflate")
    if top_grid is None or not top_grid.text:
        print("No topGridDeflate found")
        return

    # Decompress using raw deflate
    compressed = base64.b64decode(top_grid.text.strip())
    decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)

    print(f"Decompressed {len(decompressed)} bytes")

    # Analyze 2-byte patterns
    pattern_counts = Counter()
    for i in range(0, len(decompressed) - 1, 2):
        pattern = decompressed[i : i + 2]
        pattern_counts[pattern] += 1

    print("\nTop 20 2-byte patterns:")
    for pattern, count in pattern_counts.most_common(20):
        hex_pattern = pattern.hex()
        percentage = count / (len(decompressed) // 2) * 100
        print(f"  {hex_pattern}: {count:>6} occurrences ({percentage:5.1f}%)")

    # Also analyze 4-byte patterns in case terrain uses more bytes
    print("\nTop 10 4-byte patterns:")
    pattern_counts_4 = Counter()
    for i in range(0, len(decompressed) - 3, 4):
        pattern = decompressed[i : i + 4]
        pattern_counts_4[pattern] += 1

    for pattern, count in pattern_counts_4.most_common(10):
        hex_pattern = pattern.hex()
        percentage = count / (len(decompressed) // 4) * 100
        print(f"  {hex_pattern}: {count:>6} occurrences ({percentage:5.1f}%)")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    analyze_terrain_patterns(save_file)

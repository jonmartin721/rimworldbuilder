import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gzip
import base64
import zlib
from lxml import etree


def debug_terrain_data(save_path: Path):
    """Debug terrain data format in RimWorld save files"""
    print(f"Examining terrain data in: {save_path}")

    if not save_path.exists():
        print(f"Save file not found: {save_path}")
        return

    # Open and parse the save file
    with open(save_path, "rb") as f:
        magic_bytes = f.read(2)
        f.seek(0)

        if magic_bytes == b"\x1f\x8b":
            print("Detected compressed save file")
            content = gzip.decompress(f.read())
        else:
            print("Detected uncompressed save file")
            content = f.read()

    # Parse XML
    root = etree.fromstring(content)

    # Find terrain grid
    terrain_grid = root.find(".//terrainGrid")
    if terrain_grid is None:
        print("No terrainGrid found")
        return

    print("\nTerrain Grid Elements:")
    for child in terrain_grid:
        print(f"  - {child.tag}: {len(child.text) if child.text else 0} chars")

        if child.tag == "topGridDeflate" and child.text:
            print(f"    Sample data: {child.text[:100]}...")

            # Try different decompression approaches
            encoded_data = child.text.strip()

            print("\n  Trying different decompression methods:")

            try:
                # Method 1: Base64 + zlib
                compressed = base64.b64decode(encoded_data)
                print(f"    Base64 decoded size: {len(compressed)} bytes")
                print(f"    First few bytes: {compressed[:20].hex()}")

                decompressed = zlib.decompress(compressed)
                print(f"    SUCCESS zlib decompression: {len(decompressed)} bytes")
                print(f"    Sample: {decompressed[:200]}")

            except Exception as e:
                print(f"    FAILED zlib: {e}")

                try:
                    # Method 2: Raw deflate
                    decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
                    print(f"    SUCCESS Raw deflate: {len(decompressed)} bytes")
                    print(f"    Sample: {decompressed[:200]}")

                except Exception as e2:
                    print(f"    FAILED Raw deflate: {e2}")

                    try:
                        # Method 3: gzip
                        decompressed = gzip.decompress(compressed)
                        print(f"    SUCCESS gzip: {len(decompressed)} bytes")
                        print(f"    Sample: {decompressed[:200]}")

                    except Exception as e3:
                        print(f"    FAILED gzip: {e3}")

                        # Method 4: Check if it's actually uncompressed base64
                        try:
                            decoded_text = base64.b64decode(encoded_data).decode(
                                "utf-8"
                            )
                            print(
                                f"    SUCCESS Direct base64 decode: {len(decoded_text)} chars"
                            )
                            print(f"    Sample: {decoded_text[:200]}")
                        except Exception as e4:
                            print(f"    FAILED Direct base64 decode: {e4}")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    debug_terrain_data(save_file)

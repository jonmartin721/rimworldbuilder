import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib
import gzip
import bz2
import lzma


def analyze_terrain_format():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        top_grid = terrain_grid.find("topGridDeflate")
        if top_grid is not None and top_grid.text:
            data = top_grid.text.strip()

            print(f"Raw data length: {len(data)}")
            print(f"First 50 chars: {data[:50]}")
            print(f"Last 50 chars: {data[-50:]}")

            # Check if it starts with a known magic number
            print(f"\nFirst bytes as hex: {data[:8]}")

            # Try different decodings
            print("\n--- Trying different decodings ---")

            # 1. Try as custom base64 variant
            # RimWorld might use a modified base64 alphabet
            print("\n1. Checking base64 characteristics:")
            chars = set(data)
            print(f"Unique characters: {len(chars)}")
            print(
                f"Contains only base64 chars: {chars.issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='))}"
            )

            # Check for padding
            print(f"Ends with '=': {data.endswith('=')}")
            print(f"Length mod 4: {len(data) % 4}")

            # 2. Try standard base64
            try:
                decoded = base64.b64decode(data)
                print("\n2. Standard base64 decode successful!")
                print(f"Decoded length: {len(decoded)}")
                print(f"First 20 bytes: {decoded[:20]}")

                # Try different compressions on decoded data
                print("\nTrying compressions on decoded data:")

                # Try zlib
                try:
                    decompressed = zlib.decompress(decoded)
                    print("  zlib: SUCCESS")
                    return decompressed
                except zlib.error:
                    print("  zlib: failed")

                # Try gzip
                try:
                    decompressed = gzip.decompress(decoded)
                    print("  gzip: SUCCESS")
                    return decompressed
                except (OSError, gzip.BadGzipFile):
                    print("  gzip: failed")

                # Try bz2
                try:
                    decompressed = bz2.decompress(decoded)
                    print("  bz2: SUCCESS")
                    return decompressed
                except (OSError, ValueError):
                    print("  bz2: failed")

                # Try lzma
                try:
                    decompressed = lzma.decompress(decoded)
                    print("  lzma: SUCCESS")
                    return decompressed
                except lzma.LZMAError:
                    print("  lzma: failed")

                # Try raw deflate (without headers)
                try:
                    decompressed = zlib.decompress(decoded, -zlib.MAX_WBITS)
                    print("  raw deflate: SUCCESS")
                    return decompressed
                except zlib.error:
                    print("  raw deflate: failed")

            except Exception as e:
                print(f"\n2. Standard base64 decode failed: {e}")

            # 3. Try custom base64 variants
            # Sometimes games use URL-safe base64
            try:
                decoded = base64.urlsafe_b64decode(data)
                print("\n3. URL-safe base64 decode successful!")
                print(f"Decoded length: {len(decoded)}")
            except (ValueError, TypeError):
                print("\n3. URL-safe base64 failed")

            # 4. Check if it's actually readable text with some encoding
            if "HeavyBridge" in data:
                print("\n4. Data contains readable text 'HeavyBridge'")
                print("   Might be a custom text encoding or serialization")

                # Count occurrences
                count = data.count("HeavyBridge")
                print(f"   'HeavyBridge' appears {count} times")


result = analyze_terrain_format()
if result:
    print("\n=== DECOMPRESSION SUCCESSFUL ===")
    print(f"Decompressed size: {len(result)}")
    # Check for bridges
    if b"HeavyBridge" in result:
        count = result.count(b"HeavyBridge")
        print(f"Found {count} HeavyBridge references in decompressed data!")

        # Show sample
        idx = result.find(b"HeavyBridge")
        sample = result[max(0, idx - 50) : idx + 100]
        print(f"Sample around first bridge: {sample}")

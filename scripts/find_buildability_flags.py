import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import base64
import zlib


def find_buildability():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    print("Searching for buildability/affordance data...")

    # Look for various grid types that might contain buildability info
    grid_types = [
        "affordanceGrid",
        "buildabilityGrid",
        "foundationGrid",
        "terrainAffordanceGrid",
        "artificialGrid",
        "canBuildGrid",
    ]

    for grid_name in grid_types:
        grid = first_map.find(grid_name)
        if grid is not None:
            print(f"\nFound {grid_name}!")
            for child in grid:
                print(f"  Child: {child.tag}")
                if child.text and len(child.text) < 100:
                    print(f"    Text: {child.text[:50]}...")

    # Check terrainGrid for additional grids
    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        print("\nterrainGrid children:")
        for child in terrain_grid:
            print(f"  {child.tag}")

            # Check foundationGrid specifically
            if "foundation" in child.tag.lower():
                print(f"    Found foundation grid: {child.tag}")
                if child.text:
                    try:
                        # Try to decode it
                        compressed = base64.b64decode(child.text.strip())
                        decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
                        print(f"    Decompressed size: {len(decompressed)} bytes")

                        # Check if it's the same size as terrain (2 bytes per tile)
                        if len(decompressed) == 275 * 275 * 2:
                            print("    This is likely 16-bit values like terrain!")

                            # Sample the data
                            import struct

                            sample = struct.unpack("<10H", decompressed[:20])
                            print(
                                f"    First 10 values: {[f'0x{v:04X}' for v in sample]}"
                            )

                            # Check unique values
                            all_values = struct.unpack(
                                f"<{len(decompressed) // 2}H", decompressed
                            )
                            unique = set(all_values)
                            print(f"    Unique values: {len(unique)}")
                            if len(unique) < 20:
                                print(
                                    f"    Values: {[f'0x{v:04X}' for v in sorted(unique)]}"
                                )
                    except Exception as e:
                        print(f"    Could not decode: {e}")

    # Look for any element with 'build' or 'afford' in the name
    print("\n\nSearching for any buildability-related elements...")
    for elem in first_map.iter():
        tag_lower = elem.tag.lower()
        if any(
            word in tag_lower
            for word in ["build", "afford", "foundation", "artificial"]
        ):
            # Skip things we already checked
            if elem.tag not in ["canChangeTerrainOnDestroyed", "terrainGrid"]:
                print(f"Found: {elem.tag}")
                if elem.text and len(elem.text) < 200:
                    print(f"  Text: {elem.text[:100]}...")
                for child in elem:
                    if child.tag not in ["def", "id", "pos"]:
                        print(f"  Child: {child.tag}")

    # Check for edificeGrid or other building-related grids
    print("\n\nChecking for edifice/building grids...")
    edifice_grid = first_map.find("edificeGrid")
    if edifice_grid is not None:
        print("Found edificeGrid!")
        for child in edifice_grid:
            print(f"  {child.tag}")

    # Check fogGrid and other grids that might give us clues
    print("\n\nOther grids in map:")
    for child in first_map:
        if "grid" in child.tag.lower() or "Grid" in child.tag:
            if child.tag not in ["terrainGrid", "fogGrid", "tempGrid"]:
                print(f"  {child.tag}")
                # Check first level children
                child_tags = [c.tag for c in child][:5]
                if child_tags:
                    print(f"    Children: {child_tags}")


find_buildability()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree


def find_bridges_everywhere():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")

    print("Searching for completed bridges...")

    # Check terrainGrid
    terrain_grid = first_map.find("terrainGrid")
    if terrain_grid is not None:
        # Check if there's bridge terrain
        data = terrain_grid.find("data")
        if data is not None and data.text:
            if "bridge" in data.text.lower() or "Bridge" in data.text:
                print("Found 'bridge' in terrain data!")

    # Check underGrid (for buildable terrain under things)
    under_grid = first_map.find("underGrid")
    if under_grid is not None:
        data = under_grid.find("data")
        if data is not None and data.text:
            # This is compressed data but let's check the raw text
            if "bridge" in str(data.text).lower():
                print("Found 'bridge' in underGrid data!")

    # Check roofGrid
    roof_grid = first_map.find("roofGrid")
    if roof_grid is not None:
        data = roof_grid.find("data")
        if data is not None and data.text:
            if "bridge" in str(data.text).lower():
                print("Found 'bridge' in roofGrid data!")

    # Check for any element with bridge in the tag name
    for elem in first_map.iter():
        if "bridge" in elem.tag.lower():
            print(f"Found element with 'bridge' in tag: {elem.tag}")

    # Check blueprints/frames more carefully
    things = first_map.find("things")
    completed_count = 0
    frame_count = 0

    for thing in things.findall("thing"):
        def_elem = thing.find("def")
        if def_elem is not None and "Bridge" in str(def_elem.text):
            # Check if it has stuffDef (material) - completed bridges might have this
            stuff_def = thing.find("stuffDef")
            if stuff_def is not None:
                print(f"Bridge with material: {def_elem.text} made of {stuff_def.text}")
                completed_count += 1
            else:
                frame_count += 1

    print("\nSummary:")
    print(f"  Frame/Blueprint bridges: {frame_count}")
    print(f"  Bridges with materials (possibly completed): {completed_count}")

    # Look for specific bridge-floor terrain types
    print("\nSearching all map data for bridge-related strings...")
    map_str = etree.tostring(first_map, encoding="unicode")

    # Common bridge terrain def names in RimWorld
    bridge_terrain_types = [
        "BridgeHeavy",
        "Bridge_Heavy",
        "HeavyBridge",
        "BridgeSteel",
        "Bridge_Steel",
        "SteelBridge",
        "BridgeWood",
        "Bridge_Wood",
        "WoodBridge",
        "BridgeConcrete",
        "ConcreteBridge",
        "TerrainBridge",
        "Floor_Bridge",
    ]

    for bridge_type in bridge_terrain_types:
        if bridge_type in map_str:
            count = map_str.count(bridge_type)
            print(f"  Found '{bridge_type}': {count} times")


find_bridges_everywhere()

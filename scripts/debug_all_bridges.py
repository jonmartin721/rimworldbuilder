import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree


def check_all_bridges():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")
    things = first_map.find("things")

    bridge_types = {}

    for thing in things.findall("thing"):
        def_elem = thing.find("def")
        if def_elem is not None:
            def_text = def_elem.text
            # Check for anything bridge-related
            if def_text and ("bridge" in def_text.lower() or "Bridge" in def_text):
                thing_class = thing.get("Class", "Unknown")
                key = f"{def_text} (Class={thing_class})"
                bridge_types[key] = bridge_types.get(key, 0) + 1

    print("All bridge-related items found:")
    for bridge_type, count in sorted(bridge_types.items()):
        print(f"  {bridge_type}: {count}")

    # Also check for specific known bridge types
    known_types = [
        "HeavyBridge",
        "Bridge",
        "SteelBridge",
        "WoodBridge",
        "Frame_HeavyBridge",
    ]
    print("\nChecking specific types:")
    for known_type in known_types:
        count = 0
        for thing in things.findall("thing"):
            def_elem = thing.find("def")
            if def_elem is not None and def_elem.text == known_type:
                count += 1
        if count > 0:
            print(f"  {known_type}: {count}")


check_all_bridges()

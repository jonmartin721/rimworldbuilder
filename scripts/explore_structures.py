from lxml import etree
from pathlib import Path
import collections


def explore_structures(save_file):
    """Explore all building/structure types in the save file"""
    tree = etree.parse(save_file)
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")
    things = first_map.find("things")

    # Collect all definitions
    all_defs = []
    for thing in things.findall("thing"):
        def_elem = thing.find("def")
        if def_elem is not None and def_elem.text:
            all_defs.append(def_elem.text)

    # Count occurrences
    def_counts = collections.Counter(all_defs)

    # Categorize
    bridges = {}
    floors = {}
    walls = {}
    furniture = {}
    production = {}
    power = {}
    other = {}

    for def_name, count in def_counts.items():
        def_lower = def_name.lower()
        if "bridge" in def_lower:
            bridges[def_name] = count
        elif "floor" in def_lower or "tile" in def_lower or "carpet" in def_lower:
            floors[def_name] = count
        elif "wall" in def_lower:
            walls[def_name] = count
        elif (
            "bed" in def_lower
            or "table" in def_lower
            or "chair" in def_lower
            or "dresser" in def_lower
        ):
            furniture[def_name] = count
        elif (
            "bench" in def_lower
            or "stove" in def_lower
            or "brewery" in def_lower
            or "smith" in def_lower
        ):
            production[def_name] = count
        elif (
            "solar" in def_lower
            or "battery" in def_lower
            or "power" in def_lower
            or "generator" in def_lower
        ):
            power[def_name] = count
        else:
            other[def_name] = count

    print("STRUCTURE ANALYSIS")
    print("=" * 50)

    if bridges:
        print(f"\nBRIDGES ({sum(bridges.values())} total):")
        for name, count in sorted(bridges.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]:
            print(f"  {name}: {count}")

    if floors:
        print(f"\nFLOORS/TILES ({sum(floors.values())} total):")
        for name, count in sorted(floors.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]:
            print(f"  {name}: {count}")

    print(f"\nWALLS ({sum(walls.values())} total):")
    for name, count in sorted(walls.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {count}")

    print(f"\nFURNITURE ({sum(furniture.values())} total):")
    for name, count in sorted(furniture.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {count}")

    print(f"\nPRODUCTION ({sum(production.values())} total):")
    for name, count in sorted(production.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {count}")

    print(f"\nPOWER ({sum(power.values())} total):")
    for name, count in sorted(power.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {count}")

    print("\nOTHER STRUCTURES (showing items with 10+ instances):")
    for name, count in sorted(other.items(), key=lambda x: x[1], reverse=True):
        if count >= 10:
            print(f"  {name}: {count}")

    return def_counts


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    explore_structures(save_file)

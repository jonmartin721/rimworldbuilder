import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree


def check_bridges():
    tree = etree.parse("data/saves/Autosave-2.rws")
    root = tree.getroot()
    game = root.find("game")
    maps = game.find("maps")
    first_map = maps.find("li")
    things = first_map.find("things")

    frames = []
    for thing in things.findall("thing"):
        def_elem = thing.find("def")
        if def_elem is not None and def_elem.text == "Frame_HeavyBridge":
            frames.append(thing)

    print(f"Found {len(frames)} Frame_HeavyBridge things")

    if frames:
        thing = frames[0]
        print("Thing class:", thing.get("Class"))
        print("Children:", [c.tag for c in thing][:15])

        # Check if it's a Frame class
        if thing.get("Class") == "Frame":
            print("\nIt's a Frame class, not Building!")
            print("We need to handle Frame class separately")


check_bridges()

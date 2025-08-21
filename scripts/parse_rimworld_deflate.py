import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree


def parse_rimworld_deflate():
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

            # RimWorld uses a custom encoding that looks like base64 but isn't standard
            # The format is similar to yEnc encoding
            # Let's check what it actually contains
            print(f"First 100 chars of topGridDeflate: {data[:100]}")
            print(f"Length: {len(data)}")

            # Check if it's actually readable text with bridge references
            if "HeavyBridge" in data:
                print("Found HeavyBridge in raw deflate data!")
                count = data.count("HeavyBridge")
                print(f"Count: {count}")

                # Find positions
                idx = 0
                positions = []
                while True:
                    idx = data.find("HeavyBridge", idx)
                    if idx == -1:
                        break
                    positions.append(idx)
                    idx += 1

                print(f"First few positions in string: {positions[:5]}")


parse_rimworld_deflate()

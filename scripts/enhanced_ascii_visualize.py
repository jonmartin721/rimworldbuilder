import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)


def enhanced_ascii_visualize(save_file_path: Path, focus_on_bridges=False):
    """Create an enhanced ASCII visualization showing bridges and other structures"""
    parser = RimWorldSaveParser()

    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)

    if not game_state.maps:
        print("No maps found")
        return

    game_map = game_state.maps[0]
    print(f"Found {len(game_map.buildings)} buildings")
    print(
        f"Including {len([b for b in game_map.buildings if b.building_type == BuildingType.BRIDGE])} bridges"
    )

    # Find bounds
    if not game_map.buildings:
        print("No buildings to visualize")
        return

    # If focusing on bridges, center on bridge area
    if focus_on_bridges:
        bridge_positions = [
            (b.position.x, b.position.y)
            for b in game_map.buildings
            if b.building_type == BuildingType.BRIDGE
        ]
        if bridge_positions:
            x_coords, y_coords = zip(*bridge_positions)
            x_center = sum(x_coords) // len(x_coords)
            y_center = sum(y_coords) // len(y_coords)
            x_min = x_center - 40
            x_max = x_center + 40
            y_min = y_center - 25
            y_max = y_center + 25
        else:
            print("No bridges found to focus on")
            return
    else:
        x_coords = [b.position.x for b in game_map.buildings]
        y_coords = [b.position.y for b in game_map.buildings]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Limit size for ASCII display
        if x_max - x_min > 120:
            x_center = (x_min + x_max) // 2
            x_min = x_center - 60
            x_max = x_center + 60

        if y_max - y_min > 60:
            y_center = (y_min + y_max) // 2
            y_min = y_center - 30
            y_max = y_center + 30

    # Clamp to map bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(game_map.size[0], x_max)
    y_max = min(game_map.size[1], y_max)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    print(f"Showing area from ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Area size: {width}x{height}")

    # Create grid with layers
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Enhanced character mapping with priority
    char_map = {
        BuildingType.BRIDGE: "=",  # Bridges (important!)
        BuildingType.WALL: "#",
        BuildingType.DOOR: "D",
        BuildingType.FURNITURE: "F",
        BuildingType.PRODUCTION: "P",
        BuildingType.STORAGE: "S",
        BuildingType.POWER: "E",
        BuildingType.CONDUIT: "+",
        BuildingType.TEMPERATURE: "T",
        BuildingType.SECURITY: "!",
        BuildingType.FENCE: "|",
        BuildingType.LIGHT: "*",
        BuildingType.FLOOR: ".",
        BuildingType.MISC: ".",
        None: ".",
    }

    # Layer priority (lower layers get overwritten by higher layers)
    priority = {
        BuildingType.FLOOR: 1,
        BuildingType.MISC: 2,
        BuildingType.CONDUIT: 3,
        BuildingType.FENCE: 4,
        BuildingType.LIGHT: 5,
        BuildingType.STORAGE: 6,
        BuildingType.TEMPERATURE: 7,
        BuildingType.POWER: 8,
        BuildingType.PRODUCTION: 9,
        BuildingType.FURNITURE: 10,
        BuildingType.SECURITY: 11,
        BuildingType.DOOR: 12,
        BuildingType.WALL: 13,
        BuildingType.BRIDGE: 14,  # Bridges have highest priority
    }

    # Sort buildings by priority
    sorted_buildings = sorted(
        game_map.buildings, key=lambda b: priority.get(b.building_type, 0)
    )

    # Place buildings
    building_count = 0
    bridge_positions = []
    for building in sorted_buildings:
        x = building.position.x
        y = building.position.y

        if x_min <= x <= x_max and y_min <= y <= y_max:
            grid_x = x - x_min
            grid_y = height - 1 - (y - y_min)  # Flip Y axis

            if 0 <= grid_x < width and 0 <= grid_y < height:
                char = char_map.get(building.building_type, ".")
                grid[grid_y][grid_x] = char
                building_count += 1

                if building.building_type == BuildingType.BRIDGE:
                    bridge_positions.append((grid_x, grid_y))

    # Place colonists (highest priority)
    for colonist in game_map.get_colonists():
        x = colonist.position.x
        y = colonist.position.y

        if x_min <= x <= x_max and y_min <= y <= y_max:
            grid_x = x - x_min
            grid_y = height - 1 - (y - y_min)

            if 0 <= grid_x < width and 0 <= grid_y < height:
                grid[grid_y][grid_x] = "@"

    # Print grid
    print("\nEnhanced Base Layout (ASCII):")
    print("=" * (width + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("=" * (width + 2))

    print("\nLegend:")
    print("  = = Bridge (Steel Heavy Bridge)")
    print("  # = Wall")
    print("  D = Door")
    print("  F = Furniture")
    print("  P = Production")
    print("  S = Storage")
    print("  E = Power")
    print("  + = Power Conduit")
    print("  T = Temperature Control")
    print("  ! = Security")
    print("  | = Fence")
    print("  * = Light")
    print("  @ = Colonist")
    print("  . = Floor/Other")

    print(f"\nShowing {building_count} buildings in this view")
    print(f"Bridges visible: {len(bridge_positions)}")

    # Save to file
    with open("base_layout_enhanced_ascii.txt", "w") as f:
        f.write("RimWorld Base Layout - Enhanced View\n")
        f.write(f"Area: ({x_min}, {y_min}) to ({x_max}, {y_max})\n")
        f.write(
            f"Bridges: {len([b for b in game_map.buildings if b.building_type == BuildingType.BRIDGE])} total\n"
        )
        f.write("=" * (width + 2) + "\n")
        for row in grid:
            f.write("|" + "".join(row) + "|\n")
        f.write("=" * (width + 2) + "\n")
        f.write("\nLegend:\n")
        f.write("  = = Bridge (buildable area)\n")
        f.write("  # = Wall\n")
        f.write("  D = Door\n")
        f.write("  F = Furniture\n")
        f.write("  @ = Colonist\n")

    print("\nSaved as base_layout_enhanced_ascii.txt")

    # If bridges were found, show their locations
    if bridge_positions:
        print("\nBridge locations in view:")
        for i, (bx, by) in enumerate(bridge_positions[:10]):
            world_x = bx + x_min
            world_y = (height - 1 - by) + y_min
            print(f"  Bridge at grid ({bx},{by}) = world ({world_x},{world_y})")
        if len(bridge_positions) > 10:
            print(f"  ... and {len(bridge_positions) - 10} more bridges")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")

    # First show general view
    enhanced_ascii_visualize(save_file)

    # Then show bridge-focused view
    print("\n" + "=" * 60)
    print("BRIDGE-FOCUSED VIEW")
    print("=" * 60)
    enhanced_ascii_visualize(save_file, focus_on_bridges=True)

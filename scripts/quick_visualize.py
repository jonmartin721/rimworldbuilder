import matplotlib.pyplot as plt
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce logging noise


def quick_visualize(save_file_path: Path):
    """Quick visualization of base layout"""
    parser = RimWorldSaveParser()

    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)

    if not game_state.maps:
        print("No maps found")
        return

    game_map = game_state.maps[0]
    print(f"Found {len(game_map.buildings)} buildings")

    # Find the bounds of all buildings
    if not game_map.buildings:
        print("No buildings to visualize")
        return

    x_coords = [b.position.x for b in game_map.buildings]
    y_coords = [b.position.y for b in game_map.buildings]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add padding
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(game_map.size[0], x_max + padding)
    y_max = min(game_map.size[1], y_max + padding)

    width = x_max - x_min
    height = y_max - y_min

    print(f"Visualizing area from ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Area size: {width}x{height}")

    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Color mapping for building types
    colors = {
        BuildingType.WALL: "gray",
        BuildingType.DOOR: "brown",
        BuildingType.FURNITURE: "blue",
        BuildingType.PRODUCTION: "yellow",
        BuildingType.STORAGE: "green",
        BuildingType.POWER: "orange",
        BuildingType.TEMPERATURE: "cyan",
        BuildingType.SECURITY: "red",
        BuildingType.MISC: "lightgray",
        None: "lightgray",
    }

    # Plot each building as a point
    for building in game_map.buildings:
        x = building.position.x
        y = building.position.y

        if x_min <= x <= x_max and y_min <= y <= y_max:
            color = colors.get(building.building_type, "lightgray")
            # Walls get smaller markers, other buildings get larger ones
            size = 2 if building.building_type == BuildingType.WALL else 10
            ax.plot(x, y, "o", color=color, markersize=size, alpha=0.7)

    # Plot colonists with names
    for colonist in game_map.get_colonists():
        x = colonist.position.x
        y = colonist.position.y
        if x_min <= x <= x_max and y_min <= y <= y_max:
            ax.plot(x, y, "r*", markersize=15, label="Colonist")
            ax.text(x + 1, y + 1, colonist.name[:5], fontsize=8)

    # Plot zones as transparent overlays
    for zone in game_map.zones:
        zone_x = [
            cell.x
            for cell in zone.cells
            if x_min <= cell.x <= x_max and y_min <= cell.y <= y_max
        ]
        zone_y = [
            cell.y
            for cell in zone.cells
            if x_min <= cell.x <= x_max and y_min <= cell.y <= y_max
        ]

        if zone_x:
            zone_type = zone.zone_type.split("_")[-1]
            if "Growing" in zone_type:
                ax.scatter(zone_x, zone_y, c="green", alpha=0.1, s=20, marker="s")
            elif "Stockpile" in zone_type:
                ax.scatter(zone_x, zone_y, c="orange", alpha=0.1, s=20, marker="s")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"RimWorld Base Layout - {len(game_map.buildings)} buildings, {len(game_map.get_colonists())} colonists"
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", label="Walls"),
        Patch(facecolor="brown", label="Doors"),
        Patch(facecolor="blue", label="Furniture"),
        Patch(facecolor="yellow", label="Production"),
        Patch(facecolor="green", label="Storage"),
        Patch(facecolor="orange", label="Power"),
        Patch(facecolor="red", label="Security"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("base_layout_quick.png", dpi=100, bbox_inches="tight")
    print("Saved as base_layout_quick.png")

    # Print some stats
    print("\nBase Statistics:")
    print(
        f"  Walls: {len([b for b in game_map.buildings if b.building_type == BuildingType.WALL])}"
    )
    print(
        f"  Doors: {len([b for b in game_map.buildings if b.building_type == BuildingType.DOOR])}"
    )
    print(
        f"  Furniture: {len([b for b in game_map.buildings if b.building_type == BuildingType.FURNITURE])}"
    )
    print(
        f"  Power: {len([b for b in game_map.buildings if b.building_type == BuildingType.POWER])}"
    )

    plt.show()


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    quick_visualize(save_file)

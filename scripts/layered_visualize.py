import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)


def layered_visualize(save_file_path: Path, layers_to_show=None):
    """
    Create a layered visualization with toggleable features

    Args:
        save_file_path: Path to the .rws file
        layers_to_show: List of layers to display, or None for all
                       Options: 'bridges', 'floors', 'walls', 'doors', 'furniture',
                               'production', 'power', 'zones', 'colonists', 'conduits'
    """
    parser = RimWorldSaveParser()

    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)

    if not game_state.maps:
        print("No maps found")
        return

    game_map = game_state.maps[0]

    # Default to showing all layers
    if layers_to_show is None:
        layers_to_show = [
            "bridges",
            "floors",
            "walls",
            "doors",
            "furniture",
            "production",
            "power",
            "zones",
            "colonists",
        ]

    print(f"Showing layers: {', '.join(layers_to_show)}")

    # Find bounds of structures
    all_positions = []
    for building in game_map.buildings:
        all_positions.append((building.position.x, building.position.y))

    if not all_positions:
        print("No buildings found")
        return

    x_coords, y_coords = zip(*all_positions)
    x_min, x_max = min(x_coords) - 10, max(x_coords) + 10
    y_min, y_max = min(y_coords) - 10, max(y_coords) + 10

    # Clamp to map size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(game_map.size[0], x_max)
    y_max = min(game_map.size[1], y_max)

    width = x_max - x_min
    height = y_max - y_min
    scale = 6  # Pixels per tile

    print(f"Visualizing area ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Area size: {width}x{height}")

    # Create base image
    img_width = width * scale
    img_height = height * scale
    img = Image.new("RGBA", (img_width, img_height), color=(30, 30, 30, 255))

    # Create separate layers
    layers = {}

    # Zone layer (bottom)
    if "zones" in layers_to_show:
        zone_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        zone_draw = ImageDraw.Draw(zone_layer)

        for zone in game_map.zones:
            zone_type = zone.zone_type.split("_")[-1]
            for cell in zone.cells:
                if x_min <= cell.x < x_max and y_min <= cell.y < y_max:
                    x = (cell.x - x_min) * scale
                    y = (height - 1 - (cell.y - y_min)) * scale

                    if "Growing" in zone_type:
                        zone_draw.rectangle(
                            [x, y, x + scale - 1, y + scale - 1], fill=(20, 60, 20, 100)
                        )
                    elif "Stockpile" in zone_type:
                        zone_draw.rectangle(
                            [x, y, x + scale - 1, y + scale - 1], fill=(80, 50, 20, 100)
                        )
        layers["zones"] = zone_layer

    # Bridge layer (important for base planning!)
    if "bridges" in layers_to_show:
        bridge_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        bridge_draw = ImageDraw.Draw(bridge_layer)

        bridges = [
            b for b in game_map.buildings if b.building_type == BuildingType.BRIDGE
        ]
        print(f"Found {len(bridges)} bridges")

        for building in bridges:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                # Draw bridges as solid brown rectangles
                bridge_draw.rectangle(
                    [x, y, x + scale - 1, y + scale - 1],
                    fill=(101, 67, 33, 255),
                    outline=(60, 40, 20, 255),
                )
        layers["bridges"] = bridge_layer

    # Floor layer
    if "floors" in layers_to_show:
        floor_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        floor_draw = ImageDraw.Draw(floor_layer)

        floors = [
            b for b in game_map.buildings if b.building_type == BuildingType.FLOOR
        ]
        for building in floors:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                floor_draw.rectangle(
                    [x, y, x + scale - 1, y + scale - 1], fill=(80, 80, 80, 150)
                )
        layers["floors"] = floor_layer

    # Conduit layer
    if "conduits" in layers_to_show:
        conduit_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        conduit_draw = ImageDraw.Draw(conduit_layer)

        conduits = [
            b for b in game_map.buildings if b.building_type == BuildingType.CONDUIT
        ]
        print(f"Found {len(conduits)} conduits")

        for building in conduits:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale + scale // 2
                y = (height - 1 - (building.position.y - y_min)) * scale + scale // 2

                # Draw conduits as thin yellow lines
                conduit_draw.ellipse(
                    [x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 0, 150)
                )
        layers["conduits"] = conduit_layer

    # Wall layer
    if "walls" in layers_to_show:
        wall_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        wall_draw = ImageDraw.Draw(wall_layer)

        walls = [b for b in game_map.buildings if b.building_type == BuildingType.WALL]
        for building in walls:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                wall_draw.rectangle(
                    [x, y, x + scale - 1, y + scale - 1], fill=(150, 150, 150, 255)
                )
        layers["walls"] = wall_layer

    # Door layer
    if "doors" in layers_to_show:
        door_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        door_draw = ImageDraw.Draw(door_layer)

        doors = [b for b in game_map.buildings if b.building_type == BuildingType.DOOR]
        for building in doors:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                margin = scale // 4
                door_draw.rectangle(
                    [x + margin, y, x + scale - margin - 1, y + scale - 1],
                    fill=(139, 69, 19, 255),
                    outline=(80, 40, 10, 255),
                )
        layers["doors"] = door_layer

    # Furniture layer
    if "furniture" in layers_to_show:
        furniture_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        furniture_draw = ImageDraw.Draw(furniture_layer)

        furniture = [
            b for b in game_map.buildings if b.building_type == BuildingType.FURNITURE
        ]
        for building in furniture:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                margin = scale // 4
                furniture_draw.ellipse(
                    [
                        x + margin,
                        y + margin,
                        x + scale - margin - 1,
                        y + scale - margin - 1,
                    ],
                    fill=(70, 130, 180, 255),
                    outline=(40, 80, 120, 255),
                )
        layers["furniture"] = furniture_layer

    # Production layer
    if "production" in layers_to_show:
        production_layer = Image.new(
            "RGBA", (img_width, img_height), color=(0, 0, 0, 0)
        )
        production_draw = ImageDraw.Draw(production_layer)

        production = [
            b for b in game_map.buildings if b.building_type == BuildingType.PRODUCTION
        ]
        for building in production:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                production_draw.rectangle(
                    [x + 1, y + 1, x + scale - 2, y + scale - 2],
                    fill=(255, 215, 0, 255),
                    outline=(180, 150, 0, 255),
                )
        layers["production"] = production_layer

    # Power layer
    if "power" in layers_to_show:
        power_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        power_draw = ImageDraw.Draw(power_layer)

        power = [b for b in game_map.buildings if b.building_type == BuildingType.POWER]
        for building in power:
            if (
                x_min <= building.position.x < x_max
                and y_min <= building.position.y < y_max
            ):
                x = (building.position.x - x_min) * scale
                y = (height - 1 - (building.position.y - y_min)) * scale

                power_draw.rectangle(
                    [x + 1, y + 1, x + scale - 2, y + scale - 2],
                    fill=(255, 140, 0, 255),
                    outline=(180, 100, 0, 255),
                )
        layers["power"] = power_layer

    # Colonist layer (top)
    if "colonists" in layers_to_show:
        colonist_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
        colonist_draw = ImageDraw.Draw(colonist_layer)

        for colonist in game_map.get_colonists():
            if (
                x_min <= colonist.position.x < x_max
                and y_min <= colonist.position.y < y_max
            ):
                x = (colonist.position.x - x_min) * scale + scale // 2
                y = (height - 1 - (colonist.position.y - y_min)) * scale + scale // 2

                colonist_draw.ellipse(
                    [x - scale // 2, y - scale // 2, x + scale // 2, y + scale // 2],
                    fill=(255, 255, 255, 255),
                    outline=(255, 0, 0, 255),
                    width=2,
                )
        layers["colonists"] = colonist_layer

    # Composite all layers
    for layer_name, layer in layers.items():
        img = Image.alpha_composite(img, layer)

    # Add grid
    grid_layer = Image.new("RGBA", (img_width, img_height), color=(0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid_layer)
    for i in range(0, width + 1, 10):
        x = i * scale
        grid_draw.line([(x, 0), (x, img_height)], fill=(60, 60, 60, 100), width=1)
    for i in range(0, height + 1, 10):
        y = i * scale
        grid_draw.line([(0, y), (img_width, y)], fill=(60, 60, 60, 100), width=1)

    img = Image.alpha_composite(img, grid_layer)

    # Save the composite
    img.save("base_layered.png")
    print("Saved as base_layered.png")

    # Also save individual layers
    for layer_name, layer in layers.items():
        if layer_name in layers_to_show:
            layer_img = Image.new(
                "RGBA", (img_width, img_height), color=(30, 30, 30, 255)
            )
            layer_img = Image.alpha_composite(layer_img, layer)
            layer_img.save(f"base_layer_{layer_name}.png")
            print(f"Saved layer: base_layer_{layer_name}.png")

    # Print statistics
    print("\nLayer Statistics:")
    print(
        f"  Bridges: {len([b for b in game_map.buildings if b.building_type == BuildingType.BRIDGE])}"
    )
    print(
        f"  Floors: {len([b for b in game_map.buildings if b.building_type == BuildingType.FLOOR])}"
    )
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
        f"  Conduits: {len([b for b in game_map.buildings if b.building_type == BuildingType.CONDUIT])}"
    )
    print(
        f"  Power: {len([b for b in game_map.buildings if b.building_type == BuildingType.POWER])}"
    )
    print(
        f"  Production: {len([b for b in game_map.buildings if b.building_type == BuildingType.PRODUCTION])}"
    )
    print(f"  Colonists: {len(game_map.get_colonists())}")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")

    # Example: Show all layers
    layered_visualize(save_file)

    # Example: Show only bridges and walls (for base planning)
    # layered_visualize(save_file, layers_to_show=['bridges', 'walls', 'zones'])

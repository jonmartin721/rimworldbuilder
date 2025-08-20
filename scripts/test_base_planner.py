import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
from src.generators.base_planner import BasePlanner
from src.generators.room_templates import RoomType
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def test_base_planner_on_bridges(save_file_path: Path):
    """Test the base planner on bridge areas"""
    
    # Parse save file to get bridge locations
    parser = RimWorldSaveParser()
    print("Parsing save file to find bridges...")
    game_state = parser.parse_save_file(save_file_path)
    
    if not game_state.maps:
        print("No maps found")
        return
    
    game_map = game_state.maps[0]
    
    # Find all bridge positions
    bridges = [b for b in game_map.buildings if b.building_type == BuildingType.BRIDGE]
    print(f"Found {len(bridges)} heavy bridges")
    
    if not bridges:
        print("No bridges found")
        return
    
    # Get bridge positions
    bridge_positions = [(b.position.x, b.position.y) for b in bridges]
    
    # Find bounds
    x_coords = [x for x, y in bridge_positions]
    y_coords = [y for x, y in bridge_positions]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    
    print(f"Bridge area: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Grid size: {grid_width}x{grid_height}")
    
    # Convert to grid coordinates
    buildable_positions = set()
    for bx, by in bridge_positions:
        grid_x = bx - x_min
        grid_y = by - y_min
        buildable_positions.add((grid_x, grid_y))
    
    print(f"Buildable positions: {len(buildable_positions)} tiles")
    
    # Plan the base
    print("\nPlanning base layout...")
    planner = BasePlanner(grid_width, grid_height, buildable_positions, seed=42)
    
    # Define what rooms we want
    required_rooms = [
        # Essential rooms
        RoomType.KITCHEN,
        RoomType.DINING_HALL,
        RoomType.FREEZER,
        RoomType.POWER_ROOM,
        
        # Bedrooms (colonists need sleep)
        RoomType.BEDROOM_SMALL,
        RoomType.BEDROOM_SMALL,
        RoomType.BEDROOM_SMALL,
        RoomType.BEDROOM_MEDIUM,
        
        # Work areas
        RoomType.WORKSHOP,
        RoomType.STORAGE_LARGE,
        
        # Support rooms
        RoomType.HOSPITAL,
        RoomType.REC_ROOM,
    ]
    
    success = planner.plan_base(required_rooms)
    
    if success:
        print("Successfully planned base!")
        print(planner.get_summary())
        visualize_base_plan(planner, x_min, y_min, bridge_positions)
    else:
        print("Failed to plan base")


def visualize_base_plan(planner: BasePlanner, offset_x: int, offset_y: int,
                        bridge_positions: list):
    """Visualize the planned base"""
    
    scale = 10
    img_width = planner.width * scale
    img_height = planner.height * scale
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # First draw buildable areas (bridges)
    for y in range(planner.height):
        for x in range(planner.width):
            if planner.buildable_grid[y, x]:
                px = x * scale
                py = y * scale
                # Draw bridge as brown background
                draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                             fill=(101, 67, 33), outline=None)
    
    # Color mapping
    colors = {
        1: (150, 150, 150),  # Wall - gray
        2: (139, 69, 19),    # Door - brown
        3: (200, 200, 150),  # Room interior - light tan
        4: (100, 100, 100),  # Corridor - dark gray
        5: (50, 150, 50),    # Courtyard - green
    }
    
    # Draw the planned base
    for y in range(planner.height):
        for x in range(planner.width):
            cell_type = planner.grid[y, x]
            if cell_type > 0:
                px = x * scale
                py = y * scale
                color = colors.get(cell_type, (100, 100, 100))
                
                if cell_type == 1:  # Wall
                    draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                                 fill=color, outline=(0, 0, 0))
                elif cell_type == 2:  # Door
                    margin = scale // 4
                    draw.rectangle([px + margin, py, px + scale - margin - 1, py + scale - 1],
                                 fill=color, outline=(0, 0, 0))
                else:  # Floor/interior
                    margin = 1
                    draw.rectangle([px + margin, py + margin, 
                                  px + scale - margin - 1, py + scale - margin - 1],
                                 fill=color, outline=None)
    
    # Draw room labels
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 8)
    except:
        font = None
    
    for room in planner.placed_rooms:
        center_x = (room.x + room.template.width // 2) * scale
        center_y = (room.y + room.template.height // 2) * scale
        
        # Draw room type label
        room_name = room.template.room_type.value.replace('_', ' ').title()
        if font:
            draw.text((center_x, center_y), room_name[:8], 
                     fill=(255, 255, 255), font=font, anchor="mm")
        else:
            # Draw a small marker
            draw.ellipse([center_x - 2, center_y - 2, center_x + 2, center_y + 2],
                       fill=(255, 255, 255))
    
    # Add grid
    for i in range(0, planner.width + 1, 5):
        x = i * scale
        draw.line([(x, 0), (x, img_height)], fill=(60, 60, 60), width=1)
    for i in range(0, planner.height + 1, 5):
        y = i * scale
        draw.line([(0, y), (img_width, y)], fill=(60, 60, 60), width=1)
    
    # Save image
    img.save('base_planned_layout.png')
    print("Saved visualization as base_planned_layout.png")
    
    # Create a second image showing room boundaries clearly
    img2 = Image.new('RGB', (img_width, img_height), color=(30, 30, 30))
    draw2 = ImageDraw.Draw(img2)
    
    # Draw each room with a different color
    room_colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 255, 100),  # Yellow
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (255, 200, 100),  # Orange
        (200, 100, 255),  # Purple
        (100, 200, 100),  # Dark green
        (200, 200, 100),  # Olive
    ]
    
    for i, room in enumerate(planner.placed_rooms):
        color = room_colors[i % len(room_colors)]
        x1, y1, x2, y2 = room.get_bounds()
        
        # Draw room rectangle
        draw2.rectangle([x1 * scale, y1 * scale, 
                        (x2 + 1) * scale - 1, (y2 + 1) * scale - 1],
                       fill=None, outline=color, width=2)
        
        # Label the room
        center_x = (x1 + x2) // 2 * scale + scale // 2
        center_y = (y1 + y2) // 2 * scale + scale // 2
        room_name = room.template.room_type.value.replace('_', '\n')
        
        if font:
            draw2.text((center_x, center_y), room_name, 
                      fill=color, font=font, anchor="mm")
    
    img2.save('base_room_layout.png')
    print("Saved room layout as base_room_layout.png")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    
    if save_file.exists():
        test_base_planner_on_bridges(save_file)
    else:
        print("Save file not found")
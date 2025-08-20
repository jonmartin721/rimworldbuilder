from PIL import Image, ImageDraw
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)

def pil_visualize(save_file_path: Path, scale: int = 4):
    """Create a PNG visualization using PIL"""
    parser = RimWorldSaveParser()
    
    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)
    
    if not game_state.maps:
        print("No maps found")
        return
    
    game_map = game_state.maps[0]
    print(f"Found {len(game_map.buildings)} buildings")
    print(f"Found {len(game_map.get_colonists())} colonists")
    
    # Find bounds
    if not game_map.buildings:
        print("No buildings to visualize")
        return
    
    x_coords = [b.position.x for b in game_map.buildings]
    y_coords = [b.position.y for b in game_map.buildings]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(game_map.size[0], x_max + padding)
    y_max = min(game_map.size[1], y_max + padding)
    
    width = x_max - x_min
    height = y_max - y_min
    
    print(f"Visualizing area from ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Area size: {width}x{height}")
    
    # Create image
    img_width = width * scale
    img_height = height * scale
    
    # Create base image with dark background
    img = Image.new('RGB', (img_width, img_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    
    # Color mapping for building types
    colors = {
        BuildingType.WALL: (128, 128, 128),  # Gray
        BuildingType.DOOR: (139, 69, 19),    # Brown
        BuildingType.FURNITURE: (70, 130, 180),  # Steel blue
        BuildingType.PRODUCTION: (255, 215, 0),  # Gold
        BuildingType.STORAGE: (34, 139, 34),     # Forest green
        BuildingType.POWER: (255, 140, 0),       # Dark orange
        BuildingType.TEMPERATURE: (0, 191, 255), # Deep sky blue
        BuildingType.SECURITY: (220, 20, 60),    # Crimson
        BuildingType.MISC: (192, 192, 192),      # Silver
        None: (100, 100, 100)                    # Dark gray
    }
    
    # First, draw zones as background
    for zone in game_map.zones:
        zone_type = zone.zone_type.split('_')[-1]
        for cell in zone.cells:
            if x_min <= cell.x < x_max and y_min <= cell.y < y_max:
                x = (cell.x - x_min) * scale
                y = (height - 1 - (cell.y - y_min)) * scale  # Flip Y
                
                if 'Growing' in zone_type:
                    draw.rectangle([x, y, x + scale - 1, y + scale - 1], 
                                 fill=(20, 60, 20), outline=None)
                elif 'Stockpile' in zone_type:
                    draw.rectangle([x, y, x + scale - 1, y + scale - 1], 
                                 fill=(60, 40, 20), outline=None)
    
    # Draw buildings
    for building in game_map.buildings:
        if x_min <= building.position.x < x_max and y_min <= building.position.y < y_max:
            x = (building.position.x - x_min) * scale
            y = (height - 1 - (building.position.y - y_min)) * scale  # Flip Y
            
            color = colors.get(building.building_type, colors[None])
            
            # Draw building based on type
            if building.building_type == BuildingType.WALL:
                # Walls fill the whole cell
                draw.rectangle([x, y, x + scale - 1, y + scale - 1], 
                             fill=color, outline=None)
            elif building.building_type == BuildingType.DOOR:
                # Doors are smaller and centered
                margin = scale // 4
                draw.rectangle([x + margin, y, x + scale - margin - 1, y + scale - 1], 
                             fill=color, outline=(0, 0, 0))
            else:
                # Other buildings are circles
                margin = scale // 6
                draw.ellipse([x + margin, y + margin, 
                            x + scale - margin - 1, y + scale - margin - 1], 
                           fill=color, outline=(0, 0, 0))
    
    # Draw colonists as bright dots
    for colonist in game_map.get_colonists():
        if x_min <= colonist.position.x < x_max and y_min <= colonist.position.y < y_max:
            x = (colonist.position.x - x_min) * scale + scale // 2
            y = (height - 1 - (colonist.position.y - y_min)) * scale + scale // 2
            
            # Draw colonist as a bright white dot with red outline
            draw.ellipse([x - scale//2, y - scale//2, x + scale//2, y + scale//2], 
                       fill=(255, 255, 255), outline=(255, 0, 0), width=2)
    
    # Add grid lines for clarity
    for i in range(0, width + 1, 10):
        x = i * scale
        draw.line([(x, 0), (x, img_height)], fill=(60, 60, 60), width=1)
    
    for i in range(0, height + 1, 10):
        y = i * scale
        draw.line([(0, y), (img_width, y)], fill=(60, 60, 60), width=1)
    
    # Save image
    img.save('base_layout_pil.png')
    print("Saved as base_layout_pil.png")
    
    # Also create a legend
    legend_width = 200
    legend_height = len(colors) * 25 + 50
    legend = Image.new('RGB', (legend_width, legend_height), color=(255, 255, 255))
    legend_draw = ImageDraw.Draw(legend)
    
    y_offset = 10
    for building_type, color in colors.items():
        if building_type is not None:
            legend_draw.rectangle([10, y_offset, 30, y_offset + 15], fill=color, outline=(0, 0, 0))
            legend_draw.text((40, y_offset), str(building_type.value if building_type else "Unknown"), 
                           fill=(0, 0, 0))
            y_offset += 20
    
    legend.save('base_legend.png')
    print("Legend saved as base_legend.png")
    
    # Print statistics
    print(f"\nBase Statistics:")
    print(f"  Total buildings in view: {len([b for b in game_map.buildings if x_min <= b.position.x < x_max and y_min <= b.position.y < y_max])}")
    print(f"  Walls: {len([b for b in game_map.buildings if b.building_type == BuildingType.WALL])}")
    print(f"  Doors: {len([b for b in game_map.buildings if b.building_type == BuildingType.DOOR])}")
    print(f"  Furniture: {len([b for b in game_map.buildings if b.building_type == BuildingType.FURNITURE])}")
    print(f"  Production: {len([b for b in game_map.buildings if b.building_type == BuildingType.PRODUCTION])}")
    print(f"  Colonists: {len(game_map.get_colonists())}")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    pil_visualize(save_file, scale=4)
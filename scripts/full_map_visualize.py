from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)

def full_map_visualize(save_file_path: Path):
    """Create a full map visualization"""
    parser = RimWorldSaveParser()
    
    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)
    
    if not game_state.maps:
        print("No maps found")
        return
    
    game_map = game_state.maps[0]
    print(f"Map size: {game_map.size[0]}x{game_map.size[1]}")
    print(f"Found {len(game_map.buildings)} buildings")
    print(f"Found {len(game_map.get_colonists())} colonists")
    
    # Full map visualization at reduced scale
    scale = 2  # 2 pixels per tile
    width, height = game_map.size
    
    # Create image for full map
    img = Image.new('RGB', (width * scale, height * scale), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    # Color mapping
    colors = {
        BuildingType.WALL: (100, 100, 100),
        BuildingType.DOOR: (139, 90, 43),
        BuildingType.FURNITURE: (70, 130, 180),
        BuildingType.PRODUCTION: (255, 215, 0),
        BuildingType.STORAGE: (34, 139, 34),
        BuildingType.POWER: (255, 140, 0),
        BuildingType.TEMPERATURE: (0, 191, 255),
        BuildingType.SECURITY: (220, 20, 60),
        BuildingType.MISC: (60, 60, 60),
        None: (40, 40, 40)
    }
    
    # Draw zones first
    print("Drawing zones...")
    for zone in game_map.zones:
        zone_type = zone.zone_type.split('_')[-1]
        for cell in zone.cells:
            x = cell.x * scale
            y = (height - 1 - cell.y) * scale
            
            if 'Growing' in zone_type:
                draw.point((x, y), fill=(10, 40, 10))
                if scale > 1:
                    draw.point((x+1, y), fill=(10, 40, 10))
                    draw.point((x, y+1), fill=(10, 40, 10))
                    draw.point((x+1, y+1), fill=(10, 40, 10))
            elif 'Stockpile' in zone_type:
                draw.point((x, y), fill=(40, 30, 10))
                if scale > 1:
                    draw.point((x+1, y), fill=(40, 30, 10))
                    draw.point((x, y+1), fill=(40, 30, 10))
                    draw.point((x+1, y+1), fill=(40, 30, 10))
    
    # Draw all buildings
    print("Drawing buildings...")
    for building in game_map.buildings:
        x = building.position.x * scale
        y = (height - 1 - building.position.y) * scale
        
        color = colors.get(building.building_type, colors[None])
        
        # Draw pixel(s) for building
        draw.point((x, y), fill=color)
        if scale > 1:
            draw.point((x+1, y), fill=color)
            draw.point((x, y+1), fill=color)
            draw.point((x+1, y+1), fill=color)
    
    # Draw colonists as bright dots
    print("Drawing colonists...")
    for colonist in game_map.get_colonists():
        x = colonist.position.x * scale
        y = (height - 1 - colonist.position.y) * scale
        
        # Draw colonist as bright red dot
        if scale > 1:
            draw.ellipse([x-1, y-1, x+2, y+2], fill=(255, 0, 0), outline=(255, 255, 0))
        else:
            draw.point((x, y), fill=(255, 0, 0))
    
    # Save full map
    img.save('base_full_map.png')
    print("Saved as base_full_map.png")
    
    # Now create a detailed view of the main base area
    # Find the densest area
    x_coords = [b.position.x for b in game_map.buildings]
    y_coords = [b.position.y for b in game_map.buildings]
    
    if x_coords and y_coords:
        # Find center of mass
        x_center = sum(x_coords) // len(x_coords)
        y_center = sum(y_coords) // len(y_coords)
        
        # Create detailed view around center
        detail_size = 100
        detail_scale = 8
        
        x_min = max(0, x_center - detail_size // 2)
        y_min = max(0, y_center - detail_size // 2)
        x_max = min(width, x_min + detail_size)
        y_max = min(height, y_min + detail_size)
        
        print(f"\nCreating detailed view of area ({x_min}, {y_min}) to ({x_max}, {y_max})")
        
        detail_img = Image.new('RGB', (detail_size * detail_scale, detail_size * detail_scale), 
                              color=(30, 30, 30))
        detail_draw = ImageDraw.Draw(detail_img)
        
        # Draw grid
        for i in range(0, detail_size + 1, 10):
            x = i * detail_scale
            detail_draw.line([(x, 0), (x, detail_size * detail_scale)], fill=(50, 50, 50), width=1)
            detail_draw.line([(0, x), (detail_size * detail_scale, x)], fill=(50, 50, 50), width=1)
        
        # Draw zones
        for zone in game_map.zones:
            zone_type = zone.zone_type.split('_')[-1]
            for cell in zone.cells:
                if x_min <= cell.x < x_max and y_min <= cell.y < y_max:
                    x = (cell.x - x_min) * detail_scale
                    y = (detail_size - 1 - (cell.y - y_min)) * detail_scale
                    
                    if 'Growing' in zone_type:
                        detail_draw.rectangle([x, y, x + detail_scale - 1, y + detail_scale - 1], 
                                            fill=(20, 60, 20), outline=None)
                    elif 'Stockpile' in zone_type:
                        detail_draw.rectangle([x, y, x + detail_scale - 1, y + detail_scale - 1], 
                                            fill=(60, 40, 20), outline=None)
        
        # Draw buildings in detail
        building_count = 0
        for building in game_map.buildings:
            if x_min <= building.position.x < x_max and y_min <= building.position.y < y_max:
                x = (building.position.x - x_min) * detail_scale
                y = (detail_size - 1 - (building.position.y - y_min)) * detail_scale
                
                color = colors.get(building.building_type, colors[None])
                
                if building.building_type == BuildingType.WALL:
                    detail_draw.rectangle([x, y, x + detail_scale - 1, y + detail_scale - 1], 
                                        fill=color, outline=None)
                elif building.building_type == BuildingType.DOOR:
                    margin = detail_scale // 4
                    detail_draw.rectangle([x + margin, y, x + detail_scale - margin - 1, y + detail_scale - 1], 
                                        fill=color, outline=(0, 0, 0))
                else:
                    margin = detail_scale // 6
                    detail_draw.ellipse([x + margin, y + margin, 
                                       x + detail_scale - margin - 1, y + detail_scale - margin - 1], 
                                      fill=color, outline=(0, 0, 0))
                building_count += 1
        
        # Draw colonists
        for colonist in game_map.get_colonists():
            if x_min <= colonist.position.x < x_max and y_min <= colonist.position.y < y_max:
                x = (colonist.position.x - x_min) * detail_scale + detail_scale // 2
                y = (detail_size - 1 - (colonist.position.y - y_min)) * detail_scale + detail_scale // 2
                
                detail_draw.ellipse([x - detail_scale//2, y - detail_scale//2, 
                                   x + detail_scale//2, y + detail_scale//2], 
                                  fill=(255, 255, 255), outline=(255, 0, 0), width=2)
        
        detail_img.save('base_detail_view.png')
        print(f"Saved detailed view as base_detail_view.png ({building_count} buildings in view)")
    
    # Print statistics
    print(f"\nBase Statistics:")
    print(f"  Map size: {width}x{height}")
    print(f"  Total buildings: {len(game_map.buildings)}")
    print(f"  - Walls: {len([b for b in game_map.buildings if b.building_type == BuildingType.WALL])}")
    print(f"  - Doors: {len([b for b in game_map.buildings if b.building_type == BuildingType.DOOR])}")
    print(f"  - Furniture: {len([b for b in game_map.buildings if b.building_type == BuildingType.FURNITURE])}")
    print(f"  - Production: {len([b for b in game_map.buildings if b.building_type == BuildingType.PRODUCTION])}")
    print(f"  - Power: {len([b for b in game_map.buildings if b.building_type == BuildingType.POWER])}")
    print(f"  Colonists: {len(game_map.get_colonists())}")
    print(f"  Zones: {len(game_map.zones)}")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    full_map_visualize(save_file)
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.WARNING)

def text_visualize(save_file_path: Path):
    """Create a simple ASCII visualization of the base"""
    parser = RimWorldSaveParser()
    
    print("Parsing save file...")
    game_state = parser.parse_save_file(save_file_path)
    
    if not game_state.maps:
        print("No maps found")
        return
    
    game_map = game_state.maps[0]
    print(f"Found {len(game_map.buildings)} buildings")
    
    # Find bounds
    if not game_map.buildings:
        print("No buildings to visualize")
        return
    
    x_coords = [b.position.x for b in game_map.buildings]
    y_coords = [b.position.y for b in game_map.buildings]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Limit visualization size
    max_width = 100
    max_height = 50
    
    # Calculate area to show
    if x_max - x_min > max_width:
        # Focus on center area
        x_center = (x_min + x_max) // 2
        x_min = x_center - max_width // 2
        x_max = x_center + max_width // 2
    
    if y_max - y_min > max_height:
        # Focus on center area
        y_center = (y_min + y_max) // 2
        y_min = y_center - max_height // 2
        y_max = y_center + max_height // 2
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    print(f"Showing area from ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Area size: {width}x{height}")
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Building type to character mapping
    char_map = {
        BuildingType.WALL: '#',
        BuildingType.DOOR: 'D',
        BuildingType.FURNITURE: 'F',
        BuildingType.PRODUCTION: 'P',
        BuildingType.STORAGE: 'S',
        BuildingType.POWER: 'E',
        BuildingType.TEMPERATURE: 'T',
        BuildingType.SECURITY: '!',
        BuildingType.MISC: '.',
        None: '.'
    }
    
    # Place buildings
    building_count = 0
    for building in game_map.buildings:
        x = building.position.x
        y = building.position.y
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            grid_x = x - x_min
            grid_y = height - 1 - (y - y_min)  # Flip Y axis for display
            
            if 0 <= grid_x < width and 0 <= grid_y < height:
                char = char_map.get(building.building_type, '.')
                grid[grid_y][grid_x] = char
                building_count += 1
    
    # Place colonists
    for colonist in game_map.get_colonists():
        x = colonist.position.x
        y = colonist.position.y
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            grid_x = x - x_min
            grid_y = height - 1 - (y - y_min)
            
            if 0 <= grid_x < width and 0 <= grid_y < height:
                grid[grid_y][grid_x] = '@'
    
    # Print grid
    print("\nBase Layout (ASCII):")
    print("=" * (width + 2))
    for row in grid:
        print('|' + ''.join(row) + '|')
    print("=" * (width + 2))
    
    print("\nLegend:")
    print("  # = Wall")
    print("  D = Door")
    print("  F = Furniture")
    print("  P = Production")
    print("  S = Storage")
    print("  E = Power")
    print("  T = Temperature")
    print("  ! = Security")
    print("  @ = Colonist")
    print("  . = Other building")
    
    print(f"\nShowing {building_count} buildings in this view")
    
    # Save to file
    with open('base_layout_ascii.txt', 'w') as f:
        f.write(f"RimWorld Base Layout - {game_state.save_name}\n")
        f.write(f"Area: ({x_min}, {y_min}) to ({x_max}, {y_max})\n")
        f.write("=" * (width + 2) + "\n")
        for row in grid:
            f.write('|' + ''.join(row) + '|\n')
        f.write("=" * (width + 2) + "\n")
    
    print("\nSaved as base_layout_ascii.txt")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    text_visualize(save_file)
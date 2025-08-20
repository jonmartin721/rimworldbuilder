import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_base_layout(save_file_path: Path, focus_area: tuple = None):
    """
    Visualize the base layout from a RimWorld save file
    
    Args:
        save_file_path: Path to the .rws file
        focus_area: Optional (x, y, width, height) to focus on a specific area
    """
    parser = RimWorldSaveParser()
    
    print(f"Parsing save file: {save_file_path.name}")
    game_state = parser.parse_save_file(save_file_path)
    
    if not game_state.maps:
        print("No maps found in save file")
        return
    
    game_map = game_state.maps[0]
    print(f"Map size: {game_map.size[0]}x{game_map.size[1]}")
    print(f"Buildings: {len(game_map.buildings)}")
    print(f"Colonists: {len(game_map.get_colonists())}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine area to visualize
    if focus_area:
        x_start, y_start, width, height = focus_area
    else:
        # Auto-detect base area from buildings
        if game_map.buildings:
            building_positions = [b.position for b in game_map.buildings]
            x_coords = [p.x for p in building_positions]
            y_coords = [p.y for p in building_positions]
            
            x_min, x_max = min(x_coords) - 5, max(x_coords) + 5
            y_min, y_max = min(y_coords) - 5, max(y_coords) + 5
            
            x_start = max(0, x_min)
            y_start = max(0, y_min)
            width = min(game_map.size[0] - x_start, x_max - x_min)
            height = min(game_map.size[1] - y_start, y_max - y_min)
        else:
            # Default to center area
            x_start, y_start = 100, 100
            width, height = 50, 50
    
    print(f"Visualizing area: ({x_start}, {y_start}) with size {width}x{height}")
    
    # Plot 1: Building types
    building_grid = np.zeros((height, width, 3))
    
    # Define colors for different building types
    building_colors = {
        BuildingType.WALL: [0.5, 0.5, 0.5],  # Gray
        BuildingType.DOOR: [0.7, 0.5, 0.3],  # Brown
        BuildingType.FURNITURE: [0.6, 0.4, 0.2],  # Dark brown
        BuildingType.PRODUCTION: [0.8, 0.8, 0.2],  # Yellow
        BuildingType.STORAGE: [0.2, 0.6, 0.2],  # Green
        BuildingType.POWER: [0.9, 0.9, 0.1],  # Bright yellow
        BuildingType.TEMPERATURE: [0.2, 0.6, 0.9],  # Light blue
        BuildingType.SECURITY: [0.9, 0.2, 0.2],  # Red
        BuildingType.MISC: [0.7, 0.7, 0.7],  # Light gray
    }
    
    # Plot buildings
    for building in game_map.buildings:
        if (x_start <= building.position.x < x_start + width and 
            y_start <= building.position.y < y_start + height):
            
            x_idx = building.position.x - x_start
            y_idx = building.position.y - y_start
            
            color = building_colors.get(building.building_type or BuildingType.MISC, [0.7, 0.7, 0.7])
            building_grid[y_idx, x_idx] = color
    
    ax1.imshow(building_grid, origin='lower', interpolation='nearest')
    ax1.set_title('Building Types')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Add legend for building types
    legend_elements = []
    for b_type, color in building_colors.items():
        legend_elements.append(patches.Patch(color=color, label=b_type.value))
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Plot 2: Zones and colonist positions
    zone_grid = np.ones((height, width, 3))
    
    # Plot zones
    zone_colors = {
        'Growing': [0.2, 0.8, 0.2],  # Green
        'Stockpile': [0.8, 0.6, 0.2],  # Orange
        'Dumping': [0.4, 0.4, 0.4],  # Gray
    }
    
    for zone in game_map.zones:
        zone_type = zone.zone_type.split('_')[-1] if '_' in zone.zone_type else zone.zone_type
        color = zone_colors.get(zone_type, [0.5, 0.5, 0.8])
        
        for cell in zone.cells:
            if (x_start <= cell.x < x_start + width and 
                y_start <= cell.y < y_start + height):
                
                x_idx = cell.x - x_start
                y_idx = cell.y - y_start
                zone_grid[y_idx, x_idx] = color
    
    ax2.imshow(zone_grid, origin='lower', interpolation='nearest', alpha=0.7)
    
    # Plot colonist positions
    colonists = game_map.get_colonists()
    for colonist in colonists:
        if (x_start <= colonist.position.x < x_start + width and 
            y_start <= colonist.position.y < y_start + height):
            
            x_idx = colonist.position.x - x_start
            y_idx = colonist.position.y - y_start
            ax2.plot(x_idx, y_idx, 'ro', markersize=8, label=colonist.name)
    
    ax2.set_title('Zones and Colonists')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Add zone legend
    zone_legend = []
    for z_type, color in zone_colors.items():
        zone_legend.append(patches.Patch(color=color, label=z_type))
    ax2.legend(handles=zone_legend, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('base_layout.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'base_layout.png'")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("BASE ANALYSIS SUMMARY")
    print("="*50)
    
    # Calculate base metrics
    total_walls = len([b for b in game_map.buildings if b.building_type == BuildingType.WALL])
    total_doors = len([b for b in game_map.buildings if b.building_type == BuildingType.DOOR])
    total_production = len([b for b in game_map.buildings if b.building_type == BuildingType.PRODUCTION])
    
    print(f"Total Walls: {total_walls}")
    print(f"Total Doors: {total_doors}")
    print(f"Total Production Buildings: {total_production}")
    print(f"Total Colonists: {len(colonists)}")
    
    if colonists:
        print("\nColonist Positions:")
        for colonist in colonists[:5]:
            print(f"  {colonist.name}: ({colonist.position.x}, {colonist.position.y})")
        if len(colonists) > 5:
            print(f"  ... and {len(colonists) - 5} more")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    
    if save_file.exists():
        # Visualize the entire base or a specific area
        # Example: focus on area starting at (100, 100) with size 80x80
        visualize_base_layout(save_file, focus_area=(100, 100, 80, 80))
    else:
        print(f"Save file not found: {save_file}")
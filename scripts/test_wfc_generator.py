import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType
from src.generators.wfc_generator import WFCGenerator, TileType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_wfc_on_bridges(save_file_path: Path):
    """Test WFC generator on bridge areas from save file"""
    
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
    print(f"Found {len(bridges)} bridges")
    
    if not bridges:
        print("No bridges found - generating on empty area instead")
        # Test on a small area
        test_wfc_simple()
        return
    
    # Get bridge positions
    bridge_positions = [(b.position.x, b.position.y) for b in bridges]
    
    # Find bounds of bridge area
    x_coords = [x for x, y in bridge_positions]
    y_coords = [y for x, y in bridge_positions]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Create a smaller grid centered on bridges
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    
    print(f"Bridge area: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    print(f"Grid size: {grid_width}x{grid_height}")
    
    # Convert bridge positions to grid coordinates
    buildable_positions = set()
    for bx, by in bridge_positions:
        grid_x = bx - x_min
        grid_y = by - y_min
        buildable_positions.add((grid_x, grid_y))
    
    print(f"Buildable positions: {len(buildable_positions)}")
    
    # Generate base layout
    print("\nGenerating base layout with WFC...")
    generator = WFCGenerator(grid_width, grid_height, seed=42)
    
    # Add some initial constraints for a functional base
    # Place entrance at one end of bridge
    if buildable_positions:
        # Find leftmost bridge position for entrance
        entrance_x = min(x for x, y in buildable_positions)
        entrance_candidates = [(x, y) for x, y in buildable_positions if x == entrance_x]
        if entrance_candidates:
            entrance = entrance_candidates[len(entrance_candidates)//2]
            entrance_tile = generator.get_tile(entrance[0], entrance[1])
            if entrance_tile:
                entrance_tile.possible_types = {TileType.DOOR}
    
    success = generator.generate(buildable_positions)
    
    if success:
        print("Successfully generated base layout!")
        visualize_wfc_result(generator, x_min, y_min, bridge_positions)
    else:
        print("Failed to generate valid layout - trying with relaxed constraints...")
        # Try again with more relaxed constraints
        generator = WFCGenerator(grid_width, grid_height, seed=43)
        success = generator.generate(buildable_positions)
        if success:
            print("Successfully generated with relaxed constraints!")
            visualize_wfc_result(generator, x_min, y_min, bridge_positions)


def test_wfc_simple():
    """Test WFC on a simple grid without constraints"""
    print("\nTesting WFC on simple 20x20 grid...")
    
    generator = WFCGenerator(20, 20, seed=42)
    
    # Set some initial constraints to guide generation
    # Place bedroom area
    for x in range(2, 8):
        for y in range(2, 8):
            tile = generator.get_tile(x, y)
            if tile:
                tile.possible_types = {TileType.BEDROOM, TileType.WALL, TileType.DOOR}
    
    # Place workshop area
    for x in range(12, 18):
        for y in range(2, 8):
            tile = generator.get_tile(x, y)
            if tile:
                tile.possible_types = {TileType.WORKSHOP, TileType.STORAGE, TileType.WALL, TileType.DOOR}
    
    # Place kitchen/dining area
    for x in range(2, 10):
        for y in range(12, 18):
            tile = generator.get_tile(x, y)
            if tile:
                tile.possible_types = {TileType.KITCHEN, TileType.DINING, TileType.WALL, TileType.DOOR}
    
    success = generator.generate()
    
    if success:
        print("Successfully generated simple base layout!")
        visualize_wfc_result(generator, 0, 0, set())
    else:
        print("Failed to generate valid layout")


def visualize_wfc_result(generator: WFCGenerator, offset_x: int, offset_y: int, 
                         bridge_positions: set):
    """Visualize the WFC generation result"""
    
    scale = 20
    img_width = generator.width * scale
    img_height = generator.height * scale
    
    img = Image.new('RGB', (img_width, img_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Color mapping for tile types
    colors = {
        TileType.EMPTY: (50, 50, 50),
        TileType.WALL: (150, 150, 150),
        TileType.DOOR: (139, 69, 19),
        TileType.BEDROOM: (100, 150, 200),
        TileType.STORAGE: (34, 139, 34),
        TileType.WORKSHOP: (255, 215, 0),
        TileType.KITCHEN: (255, 140, 0),
        TileType.DINING: (200, 100, 50),
        TileType.RECREATION: (200, 100, 200),
        TileType.HOSPITAL: (255, 100, 100),
        TileType.RESEARCH: (100, 200, 255),
        TileType.POWER: (255, 255, 0),
        TileType.BATTERY: (200, 200, 0),
        TileType.CORRIDOR: (100, 100, 100),
        TileType.OUTDOOR: (20, 80, 20),
        TileType.FARM: (50, 150, 50),
    }
    
    # Draw tiles
    for y in range(generator.height):
        for x in range(generator.width):
            tile = generator.grid[y][x]
            px = x * scale
            py = y * scale
            
            # Draw bridge underlay if this is a bridge position
            world_x = x + offset_x
            world_y = y + offset_y
            if (world_x, world_y) in bridge_positions:
                draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                             fill=(101, 67, 33), outline=None)
            
            # Draw tile
            if tile.collapsed and tile.final_type:
                color = colors.get(tile.final_type, (100, 100, 100))
                
                if tile.final_type == TileType.WALL:
                    draw.rectangle([px, py, px + scale - 1, py + scale - 1],
                                 fill=color, outline=None)
                elif tile.final_type == TileType.DOOR:
                    margin = scale // 4
                    draw.rectangle([px + margin, py, px + scale - margin - 1, py + scale - 1],
                                 fill=color, outline=(0, 0, 0))
                elif tile.final_type in [TileType.EMPTY, TileType.OUTDOOR]:
                    # Just show the background or bridge
                    pass
                else:
                    # Draw as rounded rectangle for rooms
                    margin = 2
                    draw.rounded_rectangle([px + margin, py + margin,
                                          px + scale - margin - 1, py + scale - margin - 1],
                                         radius=3, fill=color, outline=(0, 0, 0))
    
    # Add grid
    for i in range(0, generator.width + 1):
        x = i * scale
        draw.line([(x, 0), (x, img_height)], fill=(60, 60, 60), width=1)
    for i in range(0, generator.height + 1):
        y = i * scale
        draw.line([(0, y), (img_width, y)], fill=(60, 60, 60), width=1)
    
    # Save image
    img.save('wfc_generated_base.png')
    print(f"Saved visualization as wfc_generated_base.png")
    
    # Print statistics
    tile_counts = {}
    for row in generator.grid:
        for tile in row:
            if tile.collapsed and tile.final_type:
                tile_counts[tile.final_type] = tile_counts.get(tile.final_type, 0) + 1
    
    print("\nGenerated tile statistics:")
    for tile_type, count in sorted(tile_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {tile_type.value}: {count}")


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    
    if save_file.exists():
        test_wfc_on_bridges(save_file)
    else:
        print("Save file not found - testing simple generation")
        test_wfc_simple()
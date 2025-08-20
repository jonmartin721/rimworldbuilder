import logging
from pathlib import Path
from src.parser.save_parser import RimWorldSaveParser
from src.models.game_entities import BuildingType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_save_file(file_path: Path):
    parser = RimWorldSaveParser()
    
    print(f"\n{'='*60}")
    print(f"Analyzing save file: {file_path.name}")
    print(f"{'='*60}\n")
    
    try:
        game_state = parser.parse_save_file(file_path)
        
        print(f"GAME METADATA")
        print(f"  Save Version: {game_state.save_version}")
        print(f"  Game Version: {game_state.game_version}")
        print(f"  Save Name: {game_state.save_name}")
        print(f"  Seed: {game_state.seed}")
        print(f"  Play Time: {game_state.play_time:.2f} ticks")
        print(f"  Current Tick: {game_state.tick}")
        print(f"  Date: {game_state.date}")
        
        print(f"\nMODS ({len(game_state.mod_ids)} loaded)")
        for i, mod_name in enumerate(game_state.mod_names[:10]):
            print(f"  - {mod_name}")
        if len(game_state.mod_names) > 10:
            print(f"  ... and {len(game_state.mod_names) - 10} more")
        
        print(f"\nMAPS ({len(game_state.maps)} total)")
        for i, game_map in enumerate(game_state.maps):
            print(f"\n  Map {i+1}: {game_map.map_id}")
            print(f"    Size: {game_map.size[0]}x{game_map.size[1]}")
            print(f"    Buildings: {len(game_map.buildings)}")
            print(f"    Pawns: {len(game_map.pawns)}")
            print(f"    Colonists: {len(game_map.get_colonists())}")
            print(f"    Items: {len(game_map.items)}")
            print(f"    Zones: {len(game_map.zones)}")
            print(f"    Rooms: {len(game_map.rooms)}")
            print(f"    Home Area Cells: {len(game_map.home_area)}")
            
            print(f"\n    BUILDING BREAKDOWN:")
            building_counts = {}
            for building in game_map.buildings:
                b_type = building.building_type or BuildingType.MISC
                building_counts[b_type] = building_counts.get(b_type, 0) + 1
            
            for b_type, count in sorted(building_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      {b_type.value}: {count}")
            
            if game_map.buildings:
                complete_buildings = [b for b in game_map.buildings if b.is_complete()]
                blueprint_buildings = [b for b in game_map.buildings if b.is_blueprint]
                frame_buildings = [b for b in game_map.buildings if b.is_frame]
                
                print(f"\n    CONSTRUCTION STATUS:")
                print(f"      Complete: {len(complete_buildings)}")
                print(f"      Blueprints: {len(blueprint_buildings)}")
                print(f"      Under Construction: {len(frame_buildings)}")
            
            if game_map.get_colonists():
                print(f"\n    COLONISTS:")
                for colonist in game_map.get_colonists()[:5]:
                    print(f"      - {colonist.name} (Health: {colonist.health:.0%})")
                if len(game_map.get_colonists()) > 5:
                    print(f"      ... and {len(game_map.get_colonists()) - 5} more")
            
            if game_map.zones:
                print(f"\n    ZONES:")
                zone_types = {}
                for zone in game_map.zones:
                    zone_type = zone.zone_type.split('_')[-1] if '_' in zone.zone_type else zone.zone_type
                    zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
                
                for z_type, count in sorted(zone_types.items()):
                    print(f"      {z_type}: {count}")
            
            print(f"\n    TERRAIN ANALYSIS:")
            terrain_types = {}
            buildable_count = 0
            walkable_count = 0
            
            for tile in game_map.terrain.values():
                terrain_types[tile.def_name] = terrain_types.get(tile.def_name, 0) + 1
                if tile.buildable:
                    buildable_count += 1
                if tile.walkable:
                    walkable_count += 1
            
            total_tiles = len(game_map.terrain)
            print(f"      Total Tiles: {total_tiles}")
            print(f"      Buildable: {buildable_count} ({buildable_count/max(total_tiles, 1)*100:.1f}%)")
            print(f"      Walkable: {walkable_count} ({walkable_count/max(total_tiles, 1)*100:.1f}%)")
            
            print(f"\n      Top Terrain Types:")
            for terrain_type, count in sorted(terrain_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"        {terrain_type}: {count}")
        
        print(f"\n{'='*60}")
        print(f"Successfully parsed save file!")
        print(f"{'='*60}\n")
        
        return game_state
        
    except Exception as e:
        print(f"\nError parsing save file: {e}")
        logger.exception("Failed to parse save file")
        return None


if __name__ == "__main__":
    save_file = Path("data/saves/Autosave-2.rws")
    
    if save_file.exists():
        analyze_save_file(save_file)
    else:
        print(f"Save file not found: {save_file}")
        print("Please ensure your RimWorld save file is placed in the data/saves directory")
import gzip
import io
import base64
import zlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Iterator, Tuple
from lxml import etree
import logging

from src.models.game_entities import (
    GameState, Map, Building, Pawn, Item, Zone, Room, 
    TerrainTile, Position, Material, BuildingType, ThingCategory
)

logger = logging.getLogger(__name__)


class RimWorldSaveParser:
    def __init__(self, chunk_size: int = 1024 * 1024):
        self.chunk_size = chunk_size
        self.building_type_mapping = {
            # Walls
            'Wall': BuildingType.WALL,
            'WallLamp': BuildingType.WALL,
            'VTE_WallMountedVent': BuildingType.WALL,
            
            # Doors
            'Door': BuildingType.DOOR,
            'Autodoor': BuildingType.DOOR,
            
            # Bridges and Floors
            'Bridge': BuildingType.BRIDGE,
            'HeavyBridge': BuildingType.BRIDGE,
            'Frame_HeavyBridge': BuildingType.BRIDGE,
            'Floor': BuildingType.FLOOR,
            'Tile': BuildingType.FLOOR,
            'Carpet': BuildingType.FLOOR,
            
            # Furniture
            'Bed': BuildingType.FURNITURE,
            'DoubleBed': BuildingType.FURNITURE,
            'Table': BuildingType.FURNITURE,
            'Chair': BuildingType.FURNITURE,
            'DiningChair': BuildingType.FURNITURE,
            'Dresser': BuildingType.FURNITURE,
            'EndTable': BuildingType.FURNITURE,
            'Wardrobe': BuildingType.FURNITURE,
            
            # Production
            'CraftingSpot': BuildingType.PRODUCTION,
            'ElectricSmelter': BuildingType.PRODUCTION,
            'FabricationBench': BuildingType.PRODUCTION,
            'ElectricStove': BuildingType.PRODUCTION,
            'Brewery': BuildingType.PRODUCTION,
            'ElectricTailoringBench': BuildingType.PRODUCTION,
            'ElectricSmithy': BuildingType.PRODUCTION,
            'AmmoBench': BuildingType.PRODUCTION,
            
            # Storage
            'StorageShelf': BuildingType.STORAGE,
            'Shelf': BuildingType.STORAGE,
            'Shelf_WeaponRack': BuildingType.STORAGE,
            
            # Power
            'Battery': BuildingType.POWER,
            'SolarGenerator': BuildingType.POWER,
            'PowerConduit': BuildingType.CONDUIT,
            'HiddenConduit': BuildingType.CONDUIT,
            
            # Temperature
            'Heater': BuildingType.TEMPERATURE,
            'Cooler': BuildingType.TEMPERATURE,
            'AirconIndoorUnit': BuildingType.TEMPERATURE,
            
            # Security
            'Turret': BuildingType.SECURITY,
            'Sandbags': BuildingType.SECURITY,
            'Barricade': BuildingType.SECURITY,
            
            # Fences
            'Fence': BuildingType.FENCE,
            'AncientFence': BuildingType.FENCE,
            
            # Lights
            'FloodLight': BuildingType.LIGHT,
            'StandingLamp': BuildingType.LIGHT,
        }
    
    def parse_save_file(self, file_path: Path) -> GameState:
        logger.info(f"Parsing save file: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Save file not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                magic_bytes = f.read(2)
                f.seek(0)
                
                if magic_bytes == b'\x1f\x8b':
                    logger.info("Detected compressed save file")
                    content = gzip.decompress(f.read())
                    return self._parse_xml_content(content)
                else:
                    logger.info("Detected uncompressed save file")
                    return self._parse_xml_stream(f)
        except Exception as e:
            logger.error(f"Error parsing save file: {e}")
            raise
    
    def _parse_xml_content(self, content: bytes) -> GameState:
        try:
            root = etree.fromstring(content)
            return self._extract_game_state(root)
        except etree.XMLSyntaxError as e:
            logger.error(f"XML syntax error: {e}")
            raise
    
    def _parse_xml_stream(self, stream: io.IOBase) -> GameState:
        try:
            tree = etree.parse(stream)
            root = tree.getroot()
            return self._extract_game_state(root)
        except etree.XMLSyntaxError as e:
            logger.error(f"XML syntax error: {e}")
            raise
    
    def _extract_game_state(self, root: etree.Element) -> GameState:
        meta = root.find('meta')
        game = root.find('game')
        
        # Extract basic metadata
        game_version = self._get_text(meta, 'gameVersion', 'Unknown')
        
        # Extract game info
        info = game.find('info') if game is not None else None
        
        game_state = GameState(
            save_version=game_version,
            game_version=game_version,
            save_name=self._get_text(info, 'realWorldDate', 'Unknown') if info is not None else 'Unknown',
            seed=self._get_text(info, 'seedString', '') if info is not None else '',
            play_time=float(self._get_text(info, 'realPlayTimeInteracting', '0')) if info is not None else 0,
            tick=int(self._get_text(info, 'ticksGame', '0')) if info is not None else 0,
            date=self._get_text(info, 'dateStringShort', '') if info is not None else ''
        )
        
        # Extract mod lists
        if meta is not None:
            mod_list = meta.find('modIds')
            if mod_list is not None:
                game_state.mod_ids = [li.text for li in mod_list.findall('li') if li.text]
            
            mod_names = meta.find('modNames')
            if mod_names is not None:
                game_state.mod_names = [li.text for li in mod_names.findall('li') if li.text]
        
        # Extract maps
        if game is not None:
            maps = game.find('maps')
            if maps is not None:
                for map_elem in maps.findall('li'):
                    parsed_map = self._parse_map(map_elem)
                    if parsed_map:
                        game_state.maps.append(parsed_map)
        
        return game_state
    
    def _extract_metadata(self, elem: etree.Element) -> GameState:
        meta = elem.find('.//meta')
        game = elem.find('.//game')
        
        return GameState(
            save_version=self._get_text(meta, 'gameVersion', 'Unknown'),
            game_version=self._get_text(meta, 'gameVersion', 'Unknown'),
            save_name=self._get_text(meta, 'saveName', 'Unknown'),
            seed=self._get_text(game, 'seed', ''),
            play_time=float(self._get_text(game, 'playTimeInteractions', '0')),
            tick=int(self._get_text(game, 'ticksGame', '0')),
            date=self._get_text(game, 'dateStringShort', '')
        )
    
    def _parse_map(self, map_elem: etree.Element) -> Optional[Map]:
        try:
            map_id = self._get_text(map_elem, 'uniqueID', 'unknown')
            map_size = self._parse_map_size(map_elem)
            
            game_map = Map(
                map_id=map_id,
                size=map_size
            )
            
            self._parse_terrain(map_elem, game_map)
            self._parse_things(map_elem, game_map)
            self._parse_zones(map_elem, game_map)
            self._parse_areas(map_elem, game_map)
            
            return game_map
        except Exception as e:
            logger.error(f"Error parsing map: {e}")
            return None
    
    def _parse_map_size(self, map_elem: etree.Element) -> Tuple[int, int]:
        map_info = map_elem.find('.//mapInfo')
        if map_info is not None:
            size_elem = map_info.find('.//size')
            if size_elem is not None:
                x = int(self._get_text(size_elem, 'x', '250'))
                z = int(self._get_text(size_elem, 'z', '250'))
                return (x, z)
        return (250, 250)
    
    def _parse_terrain(self, map_elem: etree.Element, game_map: Map):
        terrain_grid = map_elem.find('terrainGrid')
        if terrain_grid is None:
            return
        
        # Try different formats for terrain data
        tiles = terrain_grid.find('tiles')
        if tiles is not None and tiles.text:
            terrain_data = tiles.text.strip()
            self._decode_terrain_grid(terrain_data, game_map)
        else:
            # Try deflated format
            top_grid = terrain_grid.find('topGridDeflate')
            if top_grid is not None and top_grid.text:
                self._decode_deflated_terrain(top_grid.text, game_map)
    
    def _decode_terrain_grid(self, terrain_data: str, game_map: Map):
        lines = terrain_data.strip().split('\n')
        width, height = game_map.size
        
        for idx, line in enumerate(lines):
            if idx >= width * height:
                break
            
            x = idx % width
            y = idx // width
            
            terrain_def = line.strip()
            if terrain_def:
                tile = TerrainTile(
                    position=Position(x=x, y=y),
                    def_name=terrain_def,
                    walkable=not terrain_def.startswith('Wall'),
                    buildable='Water' not in terrain_def and 'Rock' not in terrain_def
                )
                game_map.terrain[(x, y)] = tile
    
    def _decode_deflated_terrain(self, deflated_data: str, game_map: Map):
        """Decode base64 + zlib compressed terrain data"""
        try:
            # Decode base64
            compressed = base64.b64decode(deflated_data.strip())
            # Decompress with zlib
            decompressed = zlib.decompress(compressed).decode('utf-8')
            
            # Now parse the decompressed data which is in format: terrain_def|count|terrain_def|count...
            width, height = game_map.size
            entries = decompressed.split('|')
            
            current_pos = 0
            i = 0
            while i < len(entries) - 1:
                terrain_def = entries[i]
                count = int(entries[i + 1])
                
                for _ in range(count):
                    if current_pos >= width * height:
                        break
                    
                    x = current_pos % width
                    y = current_pos // width
                    
                    if terrain_def and terrain_def != 'null':
                        tile = TerrainTile(
                            position=Position(x=x, y=y),
                            def_name=terrain_def,
                            walkable='Wall' not in terrain_def and 'Rock' not in terrain_def,
                            buildable='Water' not in terrain_def and 'Rock' not in terrain_def and 'Wall' not in terrain_def
                        )
                        game_map.terrain[(x, y)] = tile
                    
                    current_pos += 1
                
                i += 2
        except Exception as e:
            logger.warning(f"Failed to decode deflated terrain: {e}")
            # Fall back to simple parsing if decompression fails
            return
    
    def _parse_things(self, map_elem: etree.Element, game_map: Map):
        things = map_elem.find('things')
        if things is None:
            return
        
        for thing in things.findall('thing'):
            thing_class = thing.get('Class', '')
            def_name = self._get_text(thing, 'def', '')
            thing_id = self._get_text(thing, 'id', '')
            pos = self._parse_position(thing)
            
            if not pos:
                continue
            
            if 'Building' in thing_class or 'Frame' in thing_class or self._is_building(def_name):
                building = self._parse_building(thing, thing_id, def_name, pos)
                if building:
                    game_map.buildings.append(building)
            elif thing_class == 'Pawn' or 'Pawn' in def_name:
                pawn = self._parse_pawn(thing, thing_id, def_name, pos)
                if pawn:
                    game_map.pawns.append(pawn)
            else:
                item = self._parse_item(thing, thing_id, def_name, pos)
                if item:
                    game_map.items.append(item)
    
    def _is_building(self, def_name: str) -> bool:
        building_keywords = ['Wall', 'Door', 'Bed', 'Table', 'Chair', 'Bench', 
                           'Turret', 'Battery', 'Solar', 'Wind', 'Geothermal',
                           'Cooler', 'Heater', 'Storage', 'Shelf', 'Hopper']
        return any(keyword in def_name for keyword in building_keywords)
    
    def _parse_building(self, thing: etree.Element, thing_id: str, def_name: str, pos: Position) -> Optional[Building]:
        building = Building(
            id=thing_id,
            def_name=def_name,
            position=pos
        )
        
        rotation = thing.find('.//rot')
        if rotation is not None and rotation.text:
            building.rotation = int(rotation.text)
        
        stuff = thing.find('.//stuff')
        if stuff is not None and stuff.text:
            building.stuff_material = stuff.text
        
        health = thing.find('.//health')
        if health is not None and health.text:
            building.hit_points = int(health.text)
        
        # Try exact match first
        if def_name in self.building_type_mapping:
            building.building_type = self.building_type_mapping[def_name]
        else:
            # Try partial matches
            for key, b_type in self.building_type_mapping.items():
                if key in def_name:
                    building.building_type = b_type
                    break
        
        if 'Blueprint' in def_name:
            building.is_blueprint = True
        elif 'Frame' in def_name:
            building.is_frame = True
        
        return building
    
    def _parse_pawn(self, thing: etree.Element, thing_id: str, def_name: str, pos: Position) -> Optional[Pawn]:
        # Get name from various possible locations
        name = def_name
        name_elem = thing.find('name')
        if name_elem is not None:
            nick = name_elem.find('nick')
            if nick is not None and nick.text:
                name = nick.text
            elif name_elem.find('first') is not None:
                first = name_elem.find('first')
                if first is not None and first.text:
                    name = first.text
        
        pawn = Pawn(
            id=thing_id,
            name=name,
            position=pos
        )
        
        # Check faction - colonists can have various faction names starting with "Faction_"
        faction = thing.find('faction')
        if faction is not None and faction.text:
            pawn.faction = faction.text
            # Check if it's a player faction (usually starts with Faction_ followed by a number)
            # Also check for kindDef to identify colonists
            kind_def = thing.find('kindDef')
            if kind_def is not None and kind_def.text:
                if 'Colonist' in kind_def.text or 'colonist' in kind_def.text:
                    pawn.is_colonist = True
            # Also check if faction is not null and is a player faction
            if faction.text and faction.text.startswith('Faction_'):
                # Could be player faction - need to check further
                # For now, we'll consider all human pawns with factions as potential colonists
                if 'Human' in def_name:
                    pawn.is_colonist = True
        
        # Check guest status
        guest = thing.find('guest')
        if guest is not None:
            prisoner = guest.find('isPrisoner')
            if prisoner is not None and prisoner.text == 'True':
                pawn.is_prisoner = True
                pawn.is_colonist = False  # Prisoners are not colonists
        
        # Get health
        health_tracker = thing.find('healthTracker')
        if health_tracker is not None:
            summary_health = health_tracker.find('summaryHealth')
            if summary_health is not None and summary_health.text:
                try:
                    pawn.health = float(summary_health.text)
                except ValueError:
                    pawn.health = 1.0
        
        return pawn
    
    def _parse_item(self, thing: etree.Element, thing_id: str, def_name: str, pos: Position) -> Optional[Item]:
        item = Item(
            id=thing_id,
            def_name=def_name,
            position=pos
        )
        
        stack_count = thing.find('.//stackCount')
        if stack_count is not None and stack_count.text:
            item.stack_count = int(stack_count.text)
        
        quality = thing.find('.//quality')
        if quality is not None and quality.text:
            item.quality = quality.text
        
        forbidden = thing.find('.//forbidden')
        if forbidden is not None and forbidden.text == 'True':
            item.forbidden = True
        
        return item
    
    def _parse_zones(self, map_elem: etree.Element, game_map: Map):
        zone_manager = map_elem.find('zoneManager')
        if zone_manager is None:
            return
        
        all_zones = zone_manager.find('allZones')
        if all_zones is not None:
            for zone_elem in all_zones.findall('li'):
                zone = self._parse_zone(zone_elem)
                if zone:
                    game_map.zones.append(zone)
    
    def _parse_zone(self, zone_elem: etree.Element) -> Optional[Zone]:
        zone_id = self._get_text(zone_elem, 'ID', '')
        zone_label = self._get_text(zone_elem, 'label', 'Unknown Zone')
        zone_type = zone_elem.get('Class', 'Zone')
        
        zone = Zone(
            id=zone_id,
            name=zone_label,
            zone_type=zone_type,
            cells=[]
        )
        
        cells = zone_elem.find('.//cells')
        if cells is not None:
            for cell in cells.findall('li'):
                pos = self._parse_cell_position(cell.text)
                if pos:
                    zone.cells.append(pos)
        
        return zone if zone.cells else None
    
    def _parse_areas(self, map_elem: etree.Element, game_map: Map):
        area_manager = map_elem.find('areaManager')
        if area_manager is None:
            return
        
        home_area = area_manager.find('home')
        if home_area is not None:
            inner_grid = home_area.find('innerGrid')
            if inner_grid is not None and inner_grid.text:
                game_map.home_area = self._decode_area_grid(inner_grid.text, game_map.size)
    
    def _decode_area_grid(self, grid_text: str, map_size: Tuple[int, int]) -> List[Position]:
        positions = []
        width, height = map_size
        values = grid_text.strip().split(',')
        
        for idx, value in enumerate(values):
            if idx >= width * height:
                break
            
            if value.strip() == '1':
                x = idx % width
                y = idx // width
                positions.append(Position(x=x, y=y))
        
        return positions
    
    def _parse_position(self, elem: etree.Element) -> Optional[Position]:
        pos_elem = elem.find('.//pos')
        if pos_elem is not None and pos_elem.text:
            return self._parse_cell_position(pos_elem.text)
        return None
    
    def _parse_cell_position(self, text: str) -> Optional[Position]:
        if not text:
            return None
        
        parts = text.strip('()').split(',')
        if len(parts) >= 2:
            try:
                x = int(parts[0].strip())
                # In RimWorld saves, position is (x, 0, z) where z is the Y coordinate on the map
                if len(parts) >= 3:
                    y = int(parts[2].strip())  # Use the third value as Y
                else:
                    y = int(parts[1].strip())
                z = int(parts[1].strip()) if len(parts) > 2 else 0  # Middle value is elevation
                return Position(x=x, y=y, z=z)
            except ValueError:
                return None
        return None
    
    def _get_text(self, elem: Optional[etree.Element], path: str, default: str = '') -> str:
        if elem is None:
            return default
        
        if '/' in path:
            found = elem.find(path)
        else:
            found = elem.find(f'.//{path}')
        
        if found is not None and found.text:
            return found.text
        return default
"""
Extracts building, furniture, and decoration definitions from user's RimWorld mods.
This allows the ML model to learn and generate bases using actual mod content.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass, asdict
from enum import IntEnum

logger = logging.getLogger(__name__)


@dataclass
class ModItem:
    """Represents a buildable item from mods"""
    def_name: str  # e.g., "Wall_Granite", "TableLong", "PlantPot"
    label: str  # Human-readable name
    category: str  # Building, Furniture, Security, Decoration, etc.
    size: tuple  # (width, height) in tiles
    mod_source: str  # Which mod it's from
    texture_path: Optional[str] = None
    can_place_over: bool = False  # Can be placed on top of other things
    is_decoration: bool = False
    passable: bool = False
    work_type: Optional[str] = None  # For workbenches
    comfort: float = 0.0  # For furniture
    beauty: float = 0.0  # Beauty stat
    

class ExtendedCellType(IntEnum):
    """Extended cell types including mod content"""
    # Basic vanilla types (0-23)
    EMPTY = 0
    WALL = 1
    DOOR = 2
    BED = 3
    TABLE = 4
    CHAIR = 5
    STORAGE = 6
    WORKBENCH = 7
    STOVE = 8
    TORCH = 9
    FLOOR = 10
    DRESSER = 11
    ENDTABLE = 12
    RECREATION = 13
    MEDICAL_BED = 14
    RESEARCH_BENCH = 15
    BATTERY = 16
    SOLAR_PANEL = 17
    WIND_TURBINE = 18
    COOLER = 19
    HEATER = 20
    PLANT_POT = 21
    SANDBAG = 22
    TURRET = 23
    
    # Extended modded content (24-255)
    # These will be dynamically mapped from mod content
    MOD_FURNITURE_START = 24
    MOD_DECORATION_START = 50
    MOD_WORKBENCH_START = 100
    MOD_SECURITY_START = 150
    MOD_MISC_START = 200


class ModContentExtractor:
    """Extracts and categorizes content from RimWorld mods"""
    
    def __init__(self, rimworld_dir: Path = None):
        # Try to find RimWorld installation
        if rimworld_dir is None:
            self.rimworld_dir = self._find_rimworld_dir()
        else:
            self.rimworld_dir = Path(rimworld_dir)
            
        self.mod_items: Dict[str, ModItem] = {}
        self.category_mappings: Dict[str, int] = {}  # Map item names to cell type IDs
        self.next_mod_id = 24  # Start after vanilla types
        
    def _find_rimworld_dir(self) -> Path:
        """Try to find RimWorld installation directory"""
        possible_paths = [
            Path("C:/Program Files (x86)/Steam/steamapps/common/RimWorld"),
            Path("C:/Program Files/Steam/steamapps/common/RimWorld"),
            Path("D:/Steam/steamapps/common/RimWorld"),
            Path("D:/SteamLibrary/steamapps/common/RimWorld"),
            Path.home() / "Steam/steamapps/common/RimWorld"
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "Mods").exists():
                logger.info(f"Found RimWorld at: {path}")
                return path
                
        logger.warning("RimWorld installation not found. Using default content only.")
        return Path(".")
    
    def extract_from_active_mods(self, mod_config_path: Optional[Path] = None) -> Dict[str, ModItem]:
        """Extract content from user's active mods"""
        items = {}
        
        # Get active mod list from ModsConfig.xml
        if mod_config_path is None:
            config_path = Path.home() / "AppData/LocalLow/Ludeon Studios/RimWorld by Ludeon Studios/Config/ModsConfig.xml"
        else:
            config_path = mod_config_path
            
        active_mods = self._get_active_mods(config_path)
        logger.info(f"Found {len(active_mods)} active mods")
        
        # Extract from each mod
        for mod_id in active_mods:
            mod_items = self._extract_from_mod(mod_id)
            items.update(mod_items)
            
        logger.info(f"Extracted {len(items)} total items from mods")
        return items
    
    def _get_active_mods(self, config_path: Path) -> List[str]:
        """Get list of active mod IDs from ModsConfig.xml"""
        if not config_path.exists():
            logger.warning(f"ModsConfig.xml not found at {config_path}")
            return ["Core"]  # At least use Core
            
        try:
            tree = ET.parse(config_path)
            root = tree.getroot()
            
            active_mods = []
            for mod in root.findall(".//activeMods/li"):
                if mod.text:
                    active_mods.append(mod.text)
                    
            return active_mods
        except Exception as e:
            logger.error(f"Error parsing ModsConfig.xml: {e}")
            return ["Core"]
    
    def _extract_from_mod(self, mod_id: str) -> Dict[str, ModItem]:
        """Extract buildable items from a specific mod"""
        items = {}
        
        # Find mod directory
        mod_paths = [
            self.rimworld_dir / "Mods" / mod_id,
            Path.home() / f"AppData/LocalLow/Ludeon Studios/RimWorld by Ludeon Studios/Mods/{mod_id}",
            self.rimworld_dir / "Data" / mod_id  # For Core
        ]
        
        mod_dir = None
        for path in mod_paths:
            if path.exists():
                mod_dir = path
                break
                
        if not mod_dir:
            return items
            
        # Look for ThingDefs
        def_paths = list(mod_dir.glob("**/Defs/ThingDefs/*.xml"))
        def_paths.extend(list(mod_dir.glob("**/Defs/ThingDefs_*/*.xml")))
        
        for def_path in def_paths:
            try:
                tree = ET.parse(def_path)
                root = tree.getroot()
                
                for thing_def in root.findall(".//ThingDef"):
                    item = self._parse_thing_def(thing_def, mod_id)
                    if item:
                        items[item.def_name] = item
                        self._assign_cell_type(item)
                        
            except Exception as e:
                logger.debug(f"Error parsing {def_path}: {e}")
                
        return items
    
    def _parse_thing_def(self, thing_def: ET.Element, mod_id: str) -> Optional[ModItem]:
        """Parse a ThingDef XML element into a ModItem"""
        # Check if it's a buildable structure
        designator = thing_def.find(".//designationCategory")
        if designator is None:
            return None
            
        def_name = thing_def.get("Name") or thing_def.find("defName").text
        if not def_name:
            return None
            
        # Extract properties
        label = thing_def.findtext("label", def_name)
        category = designator.text
        
        # Get size
        size_elem = thing_def.find(".//size")
        if size_elem is not None:
            size = (int(size_elem.findtext("x", "1")), 
                   int(size_elem.findtext("z", "1")))
        else:
            size = (1, 1)
            
        # Check properties
        passable = thing_def.findtext(".//passability") != "Impassable"
        
        # Get stats
        beauty = 0.0
        comfort = 0.0
        stat_base = thing_def.find(".//statBases")
        if stat_base is not None:
            beauty = float(stat_base.findtext("Beauty", "0"))
            comfort = float(stat_base.findtext("Comfort", "0"))
            
        # Determine if it's decoration
        is_decoration = (
            category in ["Decoration", "Art", "Furniture"] or
            beauty > 0 or
            "sculpture" in def_name.lower() or
            "plant" in def_name.lower() or
            "pot" in def_name.lower()
        )
        
        # Get work type for workbenches
        work_type = None
        if category == "Production":
            work_giver = thing_def.findtext(".//workGiverDef")
            if work_giver:
                work_type = work_giver
                
        return ModItem(
            def_name=def_name,
            label=label,
            category=category,
            size=size,
            mod_source=mod_id,
            is_decoration=is_decoration,
            passable=passable,
            work_type=work_type,
            comfort=comfort,
            beauty=beauty
        )
    
    def _assign_cell_type(self, item: ModItem) -> int:
        """Assign a cell type ID to a mod item"""
        # Check if already assigned
        if item.def_name in self.category_mappings:
            return self.category_mappings[item.def_name]
            
        # Assign based on category
        if item.is_decoration:
            if self.next_mod_id < ExtendedCellType.MOD_WORKBENCH_START:
                cell_type = self.next_mod_id
                self.next_mod_id += 1
        elif item.category == "Production":
            base = ExtendedCellType.MOD_WORKBENCH_START
            offset = self.next_mod_id - base if self.next_mod_id >= base else 0
            cell_type = base + offset
        elif item.category == "Security":
            base = ExtendedCellType.MOD_SECURITY_START
            offset = self.next_mod_id - base if self.next_mod_id >= base else 0
            cell_type = base + offset
        else:
            cell_type = self.next_mod_id
            self.next_mod_id += 1
            
        self.category_mappings[item.def_name] = cell_type
        return cell_type
    
    def save_extracted_content(self, filepath: Path):
        """Save extracted mod content to file"""
        data = {
            'items': {k: asdict(v) for k, v in self.mod_items.items()},
            'mappings': self.category_mappings
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(self.mod_items)} mod items to {filepath}")
    
    def load_extracted_content(self, filepath: Path) -> bool:
        """Load previously extracted content"""
        if not filepath.exists():
            return False
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.mod_items = {
            k: ModItem(**v) for k, v in data['items'].items()
        }
        self.category_mappings = data['mappings']
        
        logger.info(f"Loaded {len(self.mod_items)} mod items from cache")
        return True
    
    def get_decoration_items(self) -> List[ModItem]:
        """Get all decoration items"""
        return [item for item in self.mod_items.values() if item.is_decoration]
    
    def get_furniture_items(self) -> List[ModItem]:
        """Get all furniture items"""
        return [item for item in self.mod_items.values() 
                if item.category == "Furniture" and item.comfort > 0]
    
    def get_production_items(self) -> List[ModItem]:
        """Get all production/workbench items"""
        return [item for item in self.mod_items.values() 
                if item.category == "Production"]


# Integration with ML model
def integrate_mod_content(model_dir: Path):
    """Integrate extracted mod content into ML training"""
    extractor = ModContentExtractor()
    
    # Try to load cached content first
    cache_file = model_dir / "mod_content_cache.json"
    if not extractor.load_extracted_content(cache_file):
        # Extract from active mods
        logger.info("Extracting content from active mods...")
        extractor.extract_from_active_mods()
        extractor.save_extracted_content(cache_file)
    
    # Log statistics
    decorations = extractor.get_decoration_items()
    furniture = extractor.get_furniture_items()
    production = extractor.get_production_items()
    
    logger.info("Available mod content:")
    logger.info(f"  - Decorations: {len(decorations)}")
    logger.info(f"  - Furniture: {len(furniture)}")
    logger.info(f"  - Production: {len(production)}")
    logger.info(f"  - Total items: {len(extractor.mod_items)}")
    
    # Example items
    if decorations:
        logger.info(f"Sample decorations: {[d.label for d in decorations[:5]]}")
    
    return extractor
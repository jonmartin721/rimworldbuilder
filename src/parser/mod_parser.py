"""
Parser for RimWorld mod definitions.
Extracts building definitions, materials, and properties from mod XML files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class BuildingDef:
    """Represents a building definition from a mod"""

    def_name: str
    label: str
    description: str = ""
    category: str = ""
    size: tuple = (1, 1)  # (width, height)
    passability: str = "Impassable"
    fill_percent: float = 1.0
    terrain_affordance: str = "Light"
    cost_list: Dict[str, int] = field(default_factory=dict)
    work_to_build: int = 0
    designator_dropdown: str = ""
    research_prerequisites: List[str] = field(default_factory=list)
    comfort: float = 0.0
    beauty: int = 0
    cleanliness: float = 0.0
    wealth: int = 0
    is_door: bool = False
    is_bed: bool = False
    is_table: bool = False
    is_chair: bool = False
    is_storage: bool = False
    is_production: bool = False
    is_power: bool = False
    is_joy: bool = False
    tags: Set[str] = field(default_factory=set)


class ModParser:
    """Parses RimWorld mod definition files"""

    def __init__(self, mod_path: Path):
        """
        Initialize parser with path to mod directory.

        Args:
            mod_path: Path to the mod folder (e.g., data/AlphaPrefabs)
        """
        self.mod_path = mod_path
        self.building_defs: Dict[str, BuildingDef] = {}

    def parse_all_defs(self) -> Dict[str, BuildingDef]:
        """Parse all building definitions in the mod"""
        self.building_defs.clear()

        # Look for ThingDefs in various locations
        def_paths = [
            self.mod_path / "1.5" / "Defs" / "ThingDefs_Buildings",
            self.mod_path / "1.5" / "Defs" / "ThingDefs_Items",
            self.mod_path / "1.5" / "Defs" / "ThingDefs",
            self.mod_path / "Defs" / "ThingDefs_Buildings",
            self.mod_path / "Defs" / "ThingDefs",
        ]

        for def_path in def_paths:
            if def_path.exists():
                logger.info(f"Parsing definitions from: {def_path}")
                self._parse_directory(def_path)

        logger.info(f"Parsed {len(self.building_defs)} building definitions")
        return self.building_defs

    def _parse_directory(self, directory: Path):
        """Parse all XML files in a directory"""
        for xml_file in directory.glob("*.xml"):
            try:
                self._parse_def_file(xml_file)
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")

    def _parse_def_file(self, filepath: Path):
        """Parse a single ThingDef XML file"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Find all ThingDef elements
            for thing_def in root.findall(".//ThingDef"):
                building_def = self._parse_thing_def(thing_def)
                if building_def and self._is_building_def(building_def):
                    self.building_defs[building_def.def_name] = building_def

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")

    def _parse_thing_def(self, thing_def: ET.Element) -> Optional[BuildingDef]:
        """Parse a single ThingDef element"""
        try:
            def_name = self._get_text(thing_def, "defName")
            if not def_name:
                return None

            building = BuildingDef(
                def_name=def_name,
                label=self._get_text(thing_def, "label", def_name),
                description=self._get_text(thing_def, "description", ""),
                category=self._get_text(thing_def, "category", ""),
                designator_dropdown=self._get_text(thing_def, "designatorDropdown", ""),
            )

            # Parse size
            size_elem = thing_def.find("size")
            if size_elem is not None:
                building.size = self._parse_size(size_elem.text)

            # Parse passability
            building.passability = self._get_text(
                thing_def, "passability", "Impassable"
            )
            building.fill_percent = self._get_float(thing_def, "fillPercent", 1.0)
            building.terrain_affordance = self._get_text(
                thing_def, "terrainAffordanceNeeded", "Light"
            )

            # Parse cost list
            cost_list = thing_def.find("costList")
            if cost_list is not None:
                building.cost_list = self._parse_cost_list(cost_list)

            # Parse work to build
            stat_base = thing_def.find("statBases")
            if stat_base is not None:
                building.work_to_build = self._get_float(stat_base, "WorkToBuild", 0)
                building.beauty = self._get_int(stat_base, "Beauty", 0)
                building.cleanliness = self._get_float(stat_base, "Cleanliness", 0.0)
                building.wealth = self._get_int(stat_base, "MarketValue", 0)
                building.comfort = self._get_float(stat_base, "Comfort", 0.0)

            # Parse research prerequisites
            research = thing_def.find("researchPrerequisites")
            if research is not None:
                for li in research.findall("li"):
                    if li.text:
                        building.research_prerequisites.append(li.text)

            # Detect building types
            building.is_door = self._is_door(thing_def)
            building.is_bed = self._is_bed(thing_def)
            building.is_table = self._is_table(thing_def)
            building.is_chair = self._is_chair(thing_def)
            building.is_storage = self._is_storage(thing_def)
            building.is_production = self._is_production(thing_def)
            building.is_power = self._is_power(thing_def)
            building.is_joy = self._is_joy(thing_def)

            # Extract tags
            building.tags = self._extract_tags(thing_def)

            return building

        except Exception as e:
            logger.error(f"Error parsing ThingDef: {e}")
            return None

    def _is_building_def(self, building_def: BuildingDef) -> bool:
        """Check if this is actually a building definition"""
        # Simple heuristic: has a size or is a known building type
        return (
            building_def.size != (1, 1)
            or building_def.is_door
            or building_def.is_bed
            or building_def.is_table
            or building_def.is_chair
            or building_def.is_storage
            or building_def.is_production
            or building_def.is_power
            or "Wall" in building_def.def_name
            or "Door" in building_def.def_name
            or "Conduit" in building_def.def_name
        )

    def _is_door(self, thing_def: ET.Element) -> bool:
        """Check if this is a door"""
        thing_class = self._get_text(thing_def, "thingClass", "")
        return "Door" in thing_class or "Door" in self._get_text(
            thing_def, "defName", ""
        )

    def _is_bed(self, thing_def: ET.Element) -> bool:
        """Check if this is a bed"""
        thing_class = self._get_text(thing_def, "thingClass", "")
        def_name = self._get_text(thing_def, "defName", "")
        return (
            "Bed" in thing_class
            or "Bed" in def_name
            or "Bedroll" in def_name
            or "SleepingSpot" in def_name
        )

    def _is_table(self, thing_def: ET.Element) -> bool:
        """Check if this is a table"""
        def_name = self._get_text(thing_def, "defName", "")
        return "Table" in def_name

    def _is_chair(self, thing_def: ET.Element) -> bool:
        """Check if this is a chair"""
        def_name = self._get_text(thing_def, "defName", "")
        return "Chair" in def_name or "Stool" in def_name or "Throne" in def_name

    def _is_storage(self, thing_def: ET.Element) -> bool:
        """Check if this is storage"""
        def_name = self._get_text(thing_def, "defName", "")
        thing_class = self._get_text(thing_def, "thingClass", "")
        return (
            "Storage" in thing_class
            or "Shelf" in def_name
            or "Container" in def_name
            or "Stockpile" in def_name
        )

    def _is_production(self, thing_def: ET.Element) -> bool:
        """Check if this is a production building"""
        def_name = self._get_text(thing_def, "defName", "")
        return (
            "Bench" in def_name
            or "Table_" in def_name
            or "Stove" in def_name
            or "Brewery" in def_name
            or "Loom" in def_name
            or "Smith" in def_name
        )

    def _is_power(self, thing_def: ET.Element) -> bool:
        """Check if this is power-related"""
        def_name = self._get_text(thing_def, "defName", "")
        comp_power = thing_def.find(".//CompProperties_Power")
        return (
            comp_power is not None
            or "Power" in def_name
            or "Battery" in def_name
            or "Solar" in def_name
            or "Wind" in def_name
            or "Geothermal" in def_name
        )

    def _is_joy(self, thing_def: ET.Element) -> bool:
        """Check if this is a joy/recreation building"""
        building_tags = thing_def.find("building/buildingTags")
        if building_tags is not None:
            for li in building_tags.findall("li"):
                if li.text and "Joy" in li.text:
                    return True

        def_name = self._get_text(thing_def, "defName", "")
        return (
            "Television" in def_name
            or "GameBoard" in def_name
            or "Telescope" in def_name
            or "Pool" in def_name
        )

    def _extract_tags(self, thing_def: ET.Element) -> Set[str]:
        """Extract all tags from the definition"""
        tags = set()

        # Building tags
        building_tags = thing_def.find("building/buildingTags")
        if building_tags is not None:
            for li in building_tags.findall("li"):
                if li.text:
                    tags.add(li.text)

        # Thing categories
        thing_categories = thing_def.find("thingCategories")
        if thing_categories is not None:
            for li in thing_categories.findall("li"):
                if li.text:
                    tags.add(li.text)

        return tags

    def _parse_size(self, size_str: str) -> tuple:
        """Parse size string like (2,3) to tuple"""
        if not size_str:
            return (1, 1)
        try:
            size_str = size_str.strip("()")
            parts = size_str.split(",")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
            elif len(parts) == 3:
                # (x, y, z) format - use x and z
                return (int(parts[0]), int(parts[2]))
        except (ValueError, IndexError, AttributeError):
            pass
        return (1, 1)

    def _parse_cost_list(self, cost_list: ET.Element) -> Dict[str, int]:
        """Parse cost list element"""
        costs = {}
        for child in cost_list:
            if child.text:
                try:
                    costs[child.tag] = int(child.text)
                except ValueError:
                    pass
        return costs

    def _get_text(self, element: ET.Element, path: str, default: str = "") -> str:
        """Get text from element path"""
        found = element.find(path)
        return found.text if found is not None and found.text else default

    def _get_int(self, element: ET.Element, path: str, default: int = 0) -> int:
        """Get integer from element path"""
        text = self._get_text(element, path)
        try:
            return int(float(text))
        except (ValueError, TypeError):
            return default

    def _get_float(self, element: ET.Element, path: str, default: float = 0.0) -> float:
        """Get float from element path"""
        text = self._get_text(element, path)
        try:
            return float(text)
        except (ValueError, TypeError):
            return default

    def get_buildings_by_category(self, category: str) -> List[BuildingDef]:
        """Get all buildings of a certain category"""
        return [
            b
            for b in self.building_defs.values()
            if category.lower() in b.category.lower()
        ]

    def get_buildings_by_type(
        self,
        is_door: bool = None,
        is_bed: bool = None,
        is_table: bool = None,
        is_chair: bool = None,
        is_storage: bool = None,
        is_production: bool = None,
        is_power: bool = None,
        is_joy: bool = None,
    ) -> List[BuildingDef]:
        """Get buildings matching specified type flags"""
        results = []
        for building in self.building_defs.values():
            if is_door is not None and building.is_door != is_door:
                continue
            if is_bed is not None and building.is_bed != is_bed:
                continue
            if is_table is not None and building.is_table != is_table:
                continue
            if is_chair is not None and building.is_chair != is_chair:
                continue
            if is_storage is not None and building.is_storage != is_storage:
                continue
            if is_production is not None and building.is_production != is_production:
                continue
            if is_power is not None and building.is_power != is_power:
                continue
            if is_joy is not None and building.is_joy != is_joy:
                continue
            results.append(building)
        return results

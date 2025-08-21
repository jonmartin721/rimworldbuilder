"""
Parser for AlphaPrefabs mod XML layout files.
Extracts prefab designs from the mod for pattern learning.
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass

from src.generators.prefab_analyzer import PrefabDesign

logger = logging.getLogger(__name__)


@dataclass
class AlphaPrefabLayout:
    """Represents a layout from AlphaPrefabs mod"""

    def_name: str
    width: int
    height: int
    layout_grid: list[list[str]]  # Raw item names
    terrain_grid: list[list[str]] | None = None
    roof_grid: list[list[str]] | None = None
    variations: list[str] = None


class AlphaPrefabParser:
    """Parses AlphaPrefabs mod XML files to extract layouts"""

    # Mapping of item types to numeric codes for our analyzer
    ITEM_TYPE_MAP = {
        "Wall": 1,  # Any wall type
        "Door": 2,  # Any door type
        "Bed": 3,  # Bedroom furniture
        "Bedroll": 3,
        "DoubleBed": 3,
        "Table": 4,  # Kitchen/dining furniture
        "DiningChair": 5,
        "Stool": 5,
        "ElectricStove": 4,
        "FueledStove": 4,
        "PassiveCooler": 6,  # Climate control
        "Cooler": 6,
        "Heater": 6,
        "Dresser": 3,  # Bedroom furniture
        "EndTable": 3,
        "TorchLamp": 7,  # Lighting
        "StandingLamp": 7,
        "PlantPot": 8,  # Decoration
        "Campfire": 9,  # Recreation/utility
        "AnimalSleepingBox": 10,  # Animal stuff
        ".": 0,  # Empty space
    }

    def __init__(self, alpha_prefabs_path: Path):
        """
        Initialize parser with path to AlphaPrefabs mod directory.

        Args:
            alpha_prefabs_path: Path to the AlphaPrefabs mod folder
        """
        self.mod_path = alpha_prefabs_path
        self.layouts_path = self.mod_path / "1.5" / "Defs" / "LayoutDefs"
        self.prefab_defs_path = self.mod_path / "1.5" / "Defs" / "PrefabDefs"

    def parse_all_layouts(self) -> list[AlphaPrefabLayout]:
        """Parse all layout files in the mod"""
        all_layouts = []

        if not self.layouts_path.exists():
            logger.error(f"Layouts path not found: {self.layouts_path}")
            return all_layouts

        # Parse each layout file
        for layout_file in self.layouts_path.glob("Layouts_*.xml"):
            if "Variations" in layout_file.name:
                continue  # Skip variation files for now

            logger.info(f"Parsing layout file: {layout_file.name}")
            layouts = self.parse_layout_file(layout_file)
            all_layouts.extend(layouts)

        logger.info(f"Parsed {len(all_layouts)} layouts total")
        return all_layouts

    def parse_layout_file(self, filepath: Path) -> list[AlphaPrefabLayout]:
        """Parse a single layout XML file"""
        layouts = []

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Find all StructureLayoutDef elements
            for layout_def in root.findall(".//KCSG.StructureLayoutDef"):
                layout = self._parse_layout_def(layout_def)
                if layout:
                    layouts.append(layout)

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")

        return layouts

    def _parse_layout_def(self, layout_def: ET.Element) -> AlphaPrefabLayout | None:
        """Parse a single StructureLayoutDef element"""
        try:
            def_name = layout_def.find("defName").text

            # Parse the main layout grid
            layouts_elem = layout_def.find("layouts")
            if layouts_elem is None:
                return None

            # Get the first layout (can be multiple for stacked items)
            first_layout = layouts_elem.find("li")
            if first_layout is None:
                return None

            layout_grid = []
            for row_elem in first_layout.findall("li"):
                row_text = row_elem.text
                if row_text:
                    # Split by comma to get individual items
                    row_items = [item.strip() for item in row_text.split(",")]
                    layout_grid.append(row_items)

            if not layout_grid:
                return None

            # Determine dimensions
            height = len(layout_grid)
            width = len(layout_grid[0]) if layout_grid else 0

            # Parse terrain grid if available
            terrain_grid = None
            terrain_elem = layout_def.find("terrainGrid")
            if terrain_elem is not None:
                terrain_grid = []
                for row_elem in terrain_elem.findall("li"):
                    row_text = row_elem.text
                    if row_text:
                        row_items = [item.strip() for item in row_text.split(",")]
                        terrain_grid.append(row_items)

            # Parse roof grid if available
            roof_grid = None
            roof_elem = layout_def.find("roofGrid")
            if roof_elem is not None:
                roof_grid = []
                for row_elem in roof_elem.findall("li"):
                    row_text = row_elem.text
                    if row_text:
                        row_items = [item.strip() for item in row_text.split(",")]
                        roof_grid.append(row_items)

            return AlphaPrefabLayout(
                def_name=def_name,
                width=width,
                height=height,
                layout_grid=layout_grid,
                terrain_grid=terrain_grid,
                roof_grid=roof_grid,
            )

        except Exception as e:
            logger.error(f"Error parsing layout def: {e}")
            return None

    def convert_to_prefab_design(self, layout: AlphaPrefabLayout) -> PrefabDesign:
        """Convert AlphaPrefab layout to our PrefabDesign format"""

        # Convert string grid to numeric array
        numeric_grid = np.zeros((layout.height, layout.width), dtype=int)

        for y, row in enumerate(layout.layout_grid):
            for x, item in enumerate(row):
                if item == ".":
                    numeric_grid[y, x] = 0  # Empty
                else:
                    # Try to match item type
                    item_type = self._get_item_type(item)
                    numeric_grid[y, x] = item_type

        # Determine prefab category from name
        category = self._determine_category(layout.def_name)

        return PrefabDesign(
            name=layout.def_name,
            width=layout.width,
            height=layout.height,
            layout=numeric_grid,
            metadata={
                "source": "AlphaPrefabs",
                "category": category,
                "has_terrain": layout.terrain_grid is not None,
                "has_roof": layout.roof_grid is not None,
            },
        )

    def _get_item_type(self, item_name: str) -> int:
        """Map an item name to a numeric type"""
        if item_name == ".":
            return 0

        # Check for exact matches first
        for key, value in self.ITEM_TYPE_MAP.items():
            if key in item_name:
                return value

        # Default to wall for unknown items (conservative)
        return 1

    def _determine_category(self, def_name: str) -> str:
        """Determine category from definition name"""
        name_lower = def_name.lower()

        if "bedroom" in name_lower:
            return "bedroom"
        elif "kitchen" in name_lower:
            return "kitchen"
        elif "dining" in name_lower or "eating" in name_lower:
            return "dining"
        elif "hospital" in name_lower or "medical" in name_lower:
            return "medical"
        elif "workshop" in name_lower or "production" in name_lower:
            return "workshop"
        elif "storage" in name_lower:
            return "storage"
        elif "prison" in name_lower:
            return "prison"
        elif "rec" in name_lower:
            return "recreation"
        elif "power" in name_lower:
            return "power"
        elif "research" in name_lower:
            return "research"
        elif "temple" in name_lower:
            return "temple"
        elif "tribal" in name_lower:
            return "tribal"
        else:
            return "general"

    def load_category_layouts(self, category: str) -> list[PrefabDesign]:
        """Load all layouts for a specific category"""
        all_layouts = self.parse_all_layouts()
        prefab_designs = []

        for layout in all_layouts:
            design = self.convert_to_prefab_design(layout)
            if design.metadata.get("category") == category:
                prefab_designs.append(design)

        return prefab_designs

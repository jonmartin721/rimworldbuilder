"""
Terrain analyzer for detecting buildable areas, zones, and terrain features
"""

import numpy as np
from dataclasses import dataclass
from scipy import ndimage
from enum import Enum


class TerrainType(Enum):
    """Terrain types found in RimWorld"""

    SHALLOW_WATER = "shallow_water"  # Buildable with bridges
    OCEAN = "ocean"  # Deep water - NOT buildable
    MARSH = "marsh"  # Buildable with bridges
    SAND = "sand"  # Directly buildable
    SOIL = "soil"  # Directly buildable
    RICH_SOIL = "rich_soil"  # Directly buildable, great for farming
    GRAVEL = "gravel"  # Directly buildable
    MUD = "mud"  # Buildable with bridges
    STONE = "stone"  # Directly buildable
    MOUNTAIN = "mountain"  # NOT buildable (must be mined)
    BRIDGE_HEAVY = "bridge_heavy"  # Heavy bridge over water
    BRIDGE_WOOD = "bridge_wood"  # Wood bridge over water
    UNKNOWN = "unknown"


class ZoneType(Enum):
    """Types of zones that can be identified"""

    INNER_DEFENSE = "inner_defense"  # Core protected area
    OUTER_DEFENSE = "outer_defense"  # Outer walls/perimeter
    AGRICULTURE = "agriculture"  # Farmland areas
    INDUSTRIAL = "industrial"  # Production/crafting
    RESIDENTIAL = "residential"  # Living quarters
    STORAGE = "storage"  # Stockpiles
    RECREATION = "recreation"  # Joy/social areas
    EXPANSION = "expansion"  # Future growth areas


@dataclass
class TerrainZone:
    """Represents a distinct zone on the map"""

    zone_type: ZoneType
    bounds: tuple[int, int, int, int]  # x, y, width, height
    area: int  # Total tiles
    buildable_area: int  # Buildable tiles
    terrain_composition: dict[TerrainType, int]  # Terrain type counts
    connectivity: float  # How connected this zone is (0-1)
    distance_from_center: float  # Distance from map center
    is_island: bool  # Whether completely surrounded by water
    adjacent_zones: list["TerrainZone"]  # Connected zones


@dataclass
class BuildableArea:
    """Represents a contiguous buildable area"""

    x: int
    y: int
    width: int
    height: int
    tiles: set[tuple[int, int]]
    area: int
    is_primary: bool  # Is this the main base area?
    has_water_access: bool
    perimeter_length: int
    compactness: float  # Area / perimeter ratio


class TerrainAnalyzer:
    """Analyzes terrain to identify buildable areas and optimal zones"""

    def __init__(self, map_width: int = 250, map_height: int = 250):
        self.map_width = map_width
        self.map_height = map_height
        self.terrain_grid = None
        self.foundation_grid = None
        self.buildable_mask = None
        self.water_mask = None
        self.zones = []
        self.buildable_areas = []

    def analyze_terrain(
        self, game_state, foundation_grid: np.ndarray | None = None
    ) -> dict:
        """
        Analyze terrain from game state and foundation grid
        Returns comprehensive terrain analysis
        """
        self.foundation_grid = foundation_grid

        # Create terrain masks
        self._create_terrain_masks(game_state)

        # Find buildable areas
        self.buildable_areas = self._find_buildable_areas()

        # Identify zones
        self.zones = self._identify_zones()

        # Analyze connectivity
        connectivity_map = self._analyze_connectivity()

        # Find optimal locations for different purposes
        optimal_locations = self._find_optimal_locations()

        return {
            "buildable_areas": self.buildable_areas,
            "zones": self.zones,
            "total_buildable": np.sum(self.buildable_mask),
            "total_water": np.sum(self.water_mask),
            "connectivity_map": connectivity_map,
            "optimal_locations": optimal_locations,
            "terrain_stats": self._get_terrain_stats(),
        }

    def _create_terrain_masks(self, game_state):
        """Create binary masks for different terrain types"""
        # Initialize masks - EVERYTHING is potentially buildable except ocean/mountains
        self.directly_buildable_mask = np.ones(
            (self.map_height, self.map_width), dtype=bool
        )
        self.bridge_required_mask = np.zeros(
            (self.map_height, self.map_width), dtype=bool
        )
        self.unbuildable_mask = np.zeros(
            (self.map_height, self.map_width), dtype=bool
        )  # Ocean/mountains
        self.existing_bridges_mask = np.zeros(
            (self.map_height, self.map_width), dtype=bool
        )
        self.fertile_mask = np.zeros((self.map_height, self.map_width), dtype=bool)

        # Process foundation grid if available
        if self.foundation_grid is not None:
            # 0x1A47 = Wood/Light bridge, 0x8C7D = Heavy bridge
            wood_bridges = self.foundation_grid == 0x1A47
            heavy_bridges = self.foundation_grid == 0x8C7D
            self.existing_bridges_mask = wood_bridges | heavy_bridges

            # Areas with bridges were water/marsh that required bridging
            self.bridge_required_mask[self.existing_bridges_mask] = True

            # But they're now buildable since bridges exist
            self.directly_buildable_mask[self.existing_bridges_mask] = True

        # Create combined buildable mask (direct + bridgeable)
        # Everything except ocean and mountains is buildable
        self.buildable_mask = ~self.unbuildable_mask

        # For backward compatibility
        self.water_mask = self.bridge_required_mask

        # Process terrain from game state if available
        if game_state and hasattr(game_state, "maps") and game_state.maps:
            first_map = game_state.maps[0]

            # Check for terrain data
            if hasattr(first_map, "terrain_grid"):
                # Process actual terrain data if decoded
                pass  # Would process terrain_grid here

            # NOTE: We intentionally DO NOT use existing buildings to determine buildable areas
            # All existing structures can be demolished and replaced
            # Only terrain features (water, mountains) are true constraints

    def _find_buildable_areas(self) -> list[BuildableArea]:
        """Find contiguous buildable areas using connected component analysis"""
        # Label connected components
        labeled, num_features = ndimage.label(self.buildable_mask)

        areas = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            coords = np.argwhere(mask)

            if len(coords) < 10:  # Skip tiny areas
                continue

            # Get bounding box
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            # Create tile set
            tiles = {(int(x), int(y)) for y, x in coords}

            # Calculate properties
            area = len(tiles)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Check water adjacency
            has_water = self._check_water_adjacency(tiles)

            # Calculate perimeter
            perimeter = self._calculate_perimeter(mask)

            # Calculate compactness
            compactness = area / (perimeter + 1)  # Avoid division by zero

            # Determine if primary (largest area)
            is_primary = False  # Will set after sorting

            areas.append(
                BuildableArea(
                    x=int(min_x),
                    y=int(min_y),
                    width=int(width),
                    height=int(height),
                    tiles=tiles,
                    area=area,
                    is_primary=is_primary,
                    has_water_access=has_water,
                    perimeter_length=perimeter,
                    compactness=compactness,
                )
            )

        # Sort by area and mark the largest as primary
        areas.sort(key=lambda a: a.area, reverse=True)
        if areas:
            areas[0].is_primary = True

        return areas

    def _identify_zones(self) -> list[TerrainZone]:
        """Identify different zones based on terrain and position"""
        zones = []

        if not self.buildable_areas:
            return zones

        # Get map center
        center_x, center_y = self.map_width // 2, self.map_height // 2

        for area in self.buildable_areas:
            # Calculate distance from center
            area_center_x = area.x + area.width // 2
            area_center_y = area.y + area.height // 2
            distance = np.sqrt(
                (area_center_x - center_x) ** 2 + (area_center_y - center_y) ** 2
            )

            # Determine zone type based on characteristics
            if area.is_primary:
                # Primary area - likely inner defense/core base
                zone_type = ZoneType.INNER_DEFENSE
            elif distance > self.map_width * 0.3:
                # Far from center - good for agriculture/expansion
                if area.area > 500 and area.compactness > 0.3:
                    zone_type = ZoneType.AGRICULTURE
                else:
                    zone_type = ZoneType.EXPANSION
            elif area.has_water_access:
                # Water access - good for industry/production
                zone_type = ZoneType.INDUSTRIAL
            else:
                # Default to residential/storage
                zone_type = (
                    ZoneType.RESIDENTIAL if area.area > 200 else ZoneType.STORAGE
                )

            # Analyze terrain composition
            terrain_comp = self._analyze_terrain_composition(area.tiles)

            zones.append(
                TerrainZone(
                    zone_type=zone_type,
                    bounds=(area.x, area.y, area.width, area.height),
                    area=area.area,
                    buildable_area=area.area,  # All tiles in area are buildable
                    terrain_composition=terrain_comp,
                    connectivity=area.compactness,
                    distance_from_center=distance,
                    is_island=not area.has_water_access,  # Simplified
                    adjacent_zones=[],  # Will populate later
                )
            )

        # Find adjacent zones
        self._find_adjacent_zones(zones)

        return zones

    def _check_water_adjacency(self, tiles: set[tuple[int, int]]) -> bool:
        """Check if any tiles are adjacent to water"""
        for x, y in tiles:
            # Check all adjacent tiles
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if self.water_mask[ny, nx]:
                        return True
        return False

    def _calculate_perimeter(self, mask: np.ndarray) -> int:
        """Calculate perimeter of a binary mask"""
        # Use morphological gradient to find edges
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, struct)
        perimeter_mask = mask & ~eroded
        return np.sum(perimeter_mask)

    def _analyze_terrain_composition(
        self, tiles: set[tuple[int, int]]
    ) -> dict[TerrainType, int]:
        """Analyze terrain types in given tiles"""
        composition = {}

        # For now, return simplified composition
        # In full implementation, would check actual terrain types
        composition[TerrainType.SOIL] = len(tiles)

        return composition

    def _find_adjacent_zones(self, zones: list[TerrainZone]):
        """Find which zones are adjacent to each other"""
        for i, zone1 in enumerate(zones):
            for j, zone2 in enumerate(zones[i + 1 :], i + 1):
                # Check if zones are adjacent (simplified - checking bounding box proximity)
                if self._zones_adjacent(zone1, zone2):
                    zone1.adjacent_zones.append(zone2)
                    zone2.adjacent_zones.append(zone1)

    def _zones_adjacent(
        self, zone1: TerrainZone, zone2: TerrainZone, threshold: int = 5
    ) -> bool:
        """Check if two zones are adjacent"""
        x1, y1, w1, h1 = zone1.bounds
        x2, y2, w2, h2 = zone2.bounds

        # Check if bounding boxes are close
        x_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        y_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

        return x_gap <= threshold and y_gap <= threshold

    def _analyze_connectivity(self) -> np.ndarray:
        """Create a connectivity map showing how accessible each area is"""
        # Distance transform from buildable areas
        connectivity = ndimage.distance_transform_edt(~self.buildable_mask)

        # Normalize and invert (so buildable areas have high values)
        max_dist = connectivity.max()
        if max_dist > 0:
            connectivity = 1 - (connectivity / max_dist)

        return connectivity

    def _find_optimal_locations(self) -> dict[str, tuple[int, int]]:
        """Find optimal locations for different purposes"""
        locations = {}

        if self.zones:
            # Find best location for main base (most central buildable area)
            primary_zone = next(
                (z for z in self.zones if z.zone_type == ZoneType.INNER_DEFENSE), None
            )
            if primary_zone:
                x, y, w, h = primary_zone.bounds
                locations["main_base"] = (x + w // 2, y + h // 2)

            # Find best agriculture zone (largest outer area)
            ag_zones = [z for z in self.zones if z.zone_type == ZoneType.AGRICULTURE]
            if ag_zones:
                best_ag = max(ag_zones, key=lambda z: z.area)
                x, y, w, h = best_ag.bounds
                locations["agriculture"] = (x + w // 2, y + h // 2)

            # Find best expansion area
            exp_zones = [z for z in self.zones if z.zone_type == ZoneType.EXPANSION]
            if exp_zones:
                best_exp = max(exp_zones, key=lambda z: z.area)
                x, y, w, h = best_exp.bounds
                locations["expansion"] = (x + w // 2, y + h // 2)

        return locations

    def _get_terrain_stats(self) -> dict:
        """Get statistics about the terrain"""
        return {
            "total_tiles": self.map_width * self.map_height,
            "buildable_tiles": int(np.sum(self.buildable_mask)),
            "water_tiles": int(np.sum(self.water_mask)),
            "buildable_percentage": float(
                np.sum(self.buildable_mask) / (self.map_width * self.map_height) * 100
            ),
            "num_buildable_areas": len(self.buildable_areas),
            "num_zones": len(self.zones),
            "largest_area_size": self.buildable_areas[0].area
            if self.buildable_areas
            else 0,
        }

    def get_zone_for_position(self, x: int, y: int) -> TerrainZone | None:
        """Get the zone containing the given position"""
        for zone in self.zones:
            zx, zy, zw, zh = zone.bounds
            if zx <= x < zx + zw and zy <= y < zy + zh:
                return zone
        return None

    def suggest_base_layout(self, requirements: dict) -> dict:
        """
        Suggest optimal base layout based on terrain analysis and requirements

        Args:
            requirements: Dict with keys like 'use_outer_areas', 'defense_priority', etc.

        Returns:
            Layout suggestions with zone assignments
        """
        suggestions = {"zones": [], "connections": [], "warnings": []}

        use_outer = requirements.get("use_outer_areas", False)
        requirements.get("defense_priority", "medium")
        need_agriculture = requirements.get("agriculture", False)

        # Assign inner defense zone
        inner_zones = [z for z in self.zones if z.zone_type == ZoneType.INNER_DEFENSE]
        if inner_zones:
            inner = inner_zones[0]
            suggestions["zones"].append(
                {
                    "zone": inner,
                    "purpose": "main_base",
                    "components": [
                        "bedrooms",
                        "dining",
                        "recreation",
                        "medical",
                        "storage",
                    ],
                }
            )

        # Handle outer areas if requested
        if use_outer:
            outer_zones = [
                z for z in self.zones if z.distance_from_center > self.map_width * 0.25
            ]

            for zone in outer_zones:
                if zone.zone_type == ZoneType.AGRICULTURE or (
                    need_agriculture and zone.area > 300
                ):
                    suggestions["zones"].append(
                        {
                            "zone": zone,
                            "purpose": "agriculture",
                            "components": ["farms", "greenhouses", "animal_pens"],
                        }
                    )
                elif zone.zone_type == ZoneType.EXPANSION:
                    suggestions["zones"].append(
                        {
                            "zone": zone,
                            "purpose": "outer_defense",
                            "components": ["walls", "turrets", "killboxes"],
                        }
                    )

        # Add connectivity suggestions
        for i, zone_info1 in enumerate(suggestions["zones"]):
            for zone_info2 in suggestions["zones"][i + 1 :]:
                if zone_info1["zone"] in zone_info2["zone"].adjacent_zones:
                    suggestions["connections"].append(
                        {
                            "from": zone_info1["purpose"],
                            "to": zone_info2["purpose"],
                            "type": "bridge"
                            if not zone_info1["zone"].has_water_access
                            else "path",
                        }
                    )

        # Add warnings
        if not inner_zones:
            suggestions["warnings"].append("No clear inner defense area found")

        if use_outer and not outer_zones:
            suggestions["warnings"].append(
                "No suitable outer areas found for expansion"
            )

        return suggestions

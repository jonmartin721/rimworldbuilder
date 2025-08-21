"""
Zone-aware base generator that respects terrain analysis and user zone preferences
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator
from src.analysis.terrain_analyzer import TerrainAnalyzer, ZoneType, TerrainZone
from src.models.base_grid import BaseGrid, CellType


@dataclass
class ZoneRequirements:
    """Requirements for a specific zone"""

    zone_type: ZoneType
    components: List[str]  # What to build here
    priority: int  # 1 = highest priority
    bridge_type: str  # 'heavy', 'wood', or 'auto'
    density: float  # 0.0 to 1.0, how densely to build


class ZoneAwareGenerator(EnhancedHybridGenerator):
    """
    Generator that understands terrain zones and can build appropriately

    CLEAN SLATE APPROACH:
    - ALL existing structures will be demolished/replaced
    - Designs optimal layout from scratch
    - Only respects natural terrain (water needs bridges, mountains unmovable)
    - Existing buildings are IGNORED in planning

    Features:
    - Respects terrain boundaries (water, mountains) ONLY
    - Can build bridges to expand into shallow water areas
    - Assigns different purposes to different zones
    - Supports multi-zone bases (inner defense + outer agriculture)
    """

    def __init__(self, game_state=None):
        super().__init__(game_state)
        self.terrain_analyzer = TerrainAnalyzer()
        self.terrain_analysis = None
        self.zone_assignments = {}
        self.bridge_plan = None

    def generate_with_zones(
        self, width: int, height: int, requirements: Dict, zone_preferences: Dict
    ) -> BaseGrid:
        """
        Generate a base considering terrain zones

        Args:
            width: Base width
            height: Base height
            requirements: Standard generation requirements
            zone_preferences: Dict with keys like:
                - use_outer_areas: bool
                - agriculture_zones: bool
                - defensive_layers: int (1, 2, or 3)
                - bridge_everything: bool
                - preserve_nature: bool
                - ignore_existing: bool (default True - treat as clean slate)
        """
        # Analyze terrain first
        if self.game_state:
            foundation_grid = getattr(self.game_state, "foundation_grid", None)
            self.terrain_analysis = self.terrain_analyzer.analyze_terrain(
                self.game_state, foundation_grid
            )

        # Plan zone usage based on analysis and preferences
        zone_plan = self._plan_zone_usage(requirements, zone_preferences)

        # Generate base grid
        grid = BaseGrid(width, height)

        # Apply zone-specific generation
        for zone_req in zone_plan:
            self._generate_zone(grid, zone_req, requirements)

        # Connect zones with paths/bridges
        self._connect_zones(grid, zone_plan)

        # Add defensive structures if requested
        if zone_preferences.get("defensive_layers", 1) > 1:
            self._add_outer_defenses(grid, zone_preferences["defensive_layers"])

        return grid

    def _plan_zone_usage(
        self, requirements: Dict, preferences: Dict
    ) -> List[ZoneRequirements]:
        """Plan how to use available zones"""
        plan = []

        if not self.terrain_analysis or not self.terrain_analysis["zones"]:
            # No terrain analysis - fall back to standard generation
            return [
                ZoneRequirements(
                    zone_type=ZoneType.INNER_DEFENSE,
                    components=self._get_all_components(requirements),
                    priority=1,
                    bridge_type="auto",
                    density=0.7,
                )
            ]

        zones = self.terrain_analysis["zones"]
        use_outer = preferences.get("use_outer_areas", False)
        need_agriculture = preferences.get("agriculture_zones", False)

        # Find and assign inner defense zone (main base)
        inner_zones = [z for z in zones if z.zone_type == ZoneType.INNER_DEFENSE]
        if inner_zones:
            inner = inner_zones[0]
            plan.append(
                ZoneRequirements(
                    zone_type=ZoneType.INNER_DEFENSE,
                    components=[
                        "bedrooms",
                        "medical",
                        "dining",
                        "recreation",
                        "workshop",
                        "storage",
                    ],
                    priority=1,
                    bridge_type="heavy",  # Use heavy bridges for main base
                    density=0.8,
                )
            )
            self.zone_assignments[inner] = plan[-1]

        # Handle outer areas if requested
        if use_outer:
            # Agriculture zones
            if need_agriculture:
                ag_zones = [
                    z
                    for z in zones
                    if z.zone_type == ZoneType.AGRICULTURE or z.area > 500
                ]
                for zone in ag_zones[:2]:  # Limit to 2 agriculture zones
                    plan.append(
                        ZoneRequirements(
                            zone_type=ZoneType.AGRICULTURE,
                            components=["farms", "greenhouses", "solar_panels"],
                            priority=3,
                            bridge_type="wood",  # Wood bridges sufficient for farms
                            density=0.4,
                        )
                    )
                    self.zone_assignments[zone] = plan[-1]

            # Outer defense perimeter
            outer_zones = [
                z
                for z in zones
                if z.distance_from_center > self.terrain_analyzer.map_width * 0.3
            ]
            if outer_zones and preferences.get("defensive_layers", 1) >= 2:
                # Pick zones for outer wall
                perimeter_zones = self._select_perimeter_zones(outer_zones)
                for zone in perimeter_zones:
                    plan.append(
                        ZoneRequirements(
                            zone_type=ZoneType.OUTER_DEFENSE,
                            components=["walls", "turrets", "killbox"],
                            priority=2,
                            bridge_type="heavy",
                            density=0.6,
                        )
                    )
                    self.zone_assignments[zone] = plan[-1]

        # Industrial zones near water
        if preferences.get("industrial_zones", False):
            water_zones = [
                z
                for z in zones
                if z.has_water_access and z.zone_type != ZoneType.INNER_DEFENSE
            ]
            if water_zones:
                plan.append(
                    ZoneRequirements(
                        zone_type=ZoneType.INDUSTRIAL,
                        components=["geothermal", "watermill", "production"],
                        priority=4,
                        bridge_type="heavy",
                        density=0.5,
                    )
                )
                self.zone_assignments[water_zones[0]] = plan[-1]

        return plan

    def _get_all_components(self, requirements: Dict) -> List[str]:
        """Extract all components from requirements"""
        components = []
        if "rooms" in requirements:
            for room in requirements["rooms"]:
                components.append(room.get("type", "generic"))
        return components if components else ["bedrooms", "storage", "dining"]

    def _select_perimeter_zones(self, zones: List[TerrainZone]) -> List[TerrainZone]:
        """Select zones that form a good perimeter"""
        # Simple selection - pick zones that are far from center and form a ring
        # In a real implementation, would use more sophisticated algorithm
        selected = []

        # Sort by angle from center to get zones around the perimeter
        center_x, center_y = (
            self.terrain_analyzer.map_width // 2,
            self.terrain_analyzer.map_height // 2,
        )

        for zone in zones:
            x, y, w, h = zone.bounds
            zone_center_x = x + w // 2
            zone_center_y = y + h // 2

            # Calculate angle from center
            angle = np.arctan2(zone_center_y - center_y, zone_center_x - center_x)
            zone.angle = angle  # Store for sorting

        # Sort by angle and select evenly spaced zones
        zones_sorted = sorted(zones, key=lambda z: z.angle)
        step = max(1, len(zones_sorted) // 8)  # Select up to 8 perimeter zones

        for i in range(0, len(zones_sorted), step):
            selected.append(zones_sorted[i])

        return selected

    def _generate_zone(
        self, grid: BaseGrid, zone_req: ZoneRequirements, requirements: Dict
    ):
        """Generate structures for a specific zone"""
        # Find the actual zone bounds
        zone = None
        for z, req in self.zone_assignments.items():
            if req == zone_req:
                zone = z
                break

        if not zone:
            # No specific zone - use full grid
            x, y, w, h = 0, 0, grid.width, grid.height
        else:
            x, y, w, h = zone.bounds

        # Ensure bounds are within grid
        x = max(0, min(x, grid.width - 1))
        y = max(0, min(y, grid.height - 1))
        w = min(w, grid.width - x)
        h = min(h, grid.height - y)

        # Generate based on zone type
        if zone_req.zone_type == ZoneType.INNER_DEFENSE:
            self._generate_main_base(grid, x, y, w, h, zone_req)
        elif zone_req.zone_type == ZoneType.AGRICULTURE:
            self._generate_agriculture(grid, x, y, w, h, zone_req)
        elif zone_req.zone_type == ZoneType.OUTER_DEFENSE:
            self._generate_defenses(grid, x, y, w, h, zone_req)
        elif zone_req.zone_type == ZoneType.INDUSTRIAL:
            self._generate_industrial(grid, x, y, w, h, zone_req)

    def _generate_main_base(
        self, grid: BaseGrid, x: int, y: int, w: int, h: int, zone_req: ZoneRequirements
    ):
        """Generate main base structures"""
        # Place walls around perimeter
        for i in range(x, x + w):
            if 0 <= i < grid.width:
                if 0 <= y < grid.height:
                    grid.set_cell(i, y, CellType.WALL)
                if 0 <= y + h - 1 < grid.height:
                    grid.set_cell(i, y + h - 1, CellType.WALL)

        for j in range(y, y + h):
            if 0 <= j < grid.height:
                if 0 <= x < grid.width:
                    grid.set_cell(x, j, CellType.WALL)
                if 0 <= x + w - 1 < grid.width:
                    grid.set_cell(x + w - 1, j, CellType.WALL)

        # Add entrance
        entrance_x = x + w // 2
        if 0 <= entrance_x < grid.width and 0 <= y < grid.height:
            grid.set_cell(entrance_x, y, CellType.DOOR)

        # Fill interior with rooms (simplified)
        room_size = 7
        for rx in range(x + 2, x + w - room_size, room_size + 1):
            for ry in range(y + 2, y + h - room_size, room_size + 1):
                self._place_room(grid, rx, ry, room_size, room_size)

    def _generate_agriculture(
        self, grid: BaseGrid, x: int, y: int, w: int, h: int, zone_req: ZoneRequirements
    ):
        """Generate agricultural areas"""
        # Create farm plots
        plot_size = 11  # Standard RimWorld growing zone size

        for fx in range(x, x + w - plot_size, plot_size + 2):
            for fy in range(y, y + h - plot_size, plot_size + 2):
                # Place growing zone
                for i in range(fx, min(fx + plot_size, x + w)):
                    for j in range(fy, min(fy + plot_size, y + h)):
                        if 0 <= i < grid.width and 0 <= j < grid.height:
                            grid.set_cell(i, j, CellType.GROWING_ZONE)

        # Add sun lamps at regular intervals
        lamp_spacing = 13  # Optimal sun lamp spacing
        for lx in range(x + lamp_spacing // 2, x + w, lamp_spacing):
            for ly in range(y + lamp_spacing // 2, y + h, lamp_spacing):
                if 0 <= lx < grid.width and 0 <= ly < grid.height:
                    grid.set_cell(lx, ly, CellType.FURNITURE)  # Sun lamp

    def _generate_defenses(
        self, grid: BaseGrid, x: int, y: int, w: int, h: int, zone_req: ZoneRequirements
    ):
        """Generate defensive structures"""
        # Create defensive wall with turrets
        turret_spacing = 8

        # Place walls
        for i in range(x, x + w):
            if 0 <= i < grid.width:
                if 0 <= y < grid.height:
                    grid.set_cell(i, y, CellType.WALL)
                    # Add turrets at intervals
                    if (i - x) % turret_spacing == 0:
                        if y + 1 < grid.height:
                            grid.set_cell(i, y + 1, CellType.DEFENSE)

    def _generate_industrial(
        self, grid: BaseGrid, x: int, y: int, w: int, h: int, zone_req: ZoneRequirements
    ):
        """Generate industrial/production areas"""
        # Place production buildings
        building_size = 5

        for bx in range(x, x + w - building_size, building_size + 2):
            for by in range(y, y + h - building_size, building_size + 2):
                self._place_room(grid, bx, by, building_size, building_size)
                # Mark center as production
                cx, cy = bx + building_size // 2, by + building_size // 2
                if 0 <= cx < grid.width and 0 <= cy < grid.height:
                    grid.set_cell(cx, cy, CellType.PRODUCTION)

    def _place_room(self, grid: BaseGrid, x: int, y: int, w: int, h: int):
        """Place a basic room"""
        # Walls
        for i in range(x, min(x + w, grid.width)):
            if 0 <= y < grid.height:
                grid.set_cell(i, y, CellType.WALL)
            if 0 <= y + h - 1 < grid.height:
                grid.set_cell(i, y + h - 1, CellType.WALL)

        for j in range(y, min(y + h, grid.height)):
            if 0 <= x < grid.width:
                grid.set_cell(x, j, CellType.WALL)
            if 0 <= x + w - 1 < grid.width:
                grid.set_cell(x + w - 1, j, CellType.WALL)

        # Door
        if 0 <= x + w // 2 < grid.width and 0 <= y < grid.height:
            grid.set_cell(x + w // 2, y, CellType.DOOR)

        # Floor
        for i in range(x + 1, min(x + w - 1, grid.width - 1)):
            for j in range(y + 1, min(y + h - 1, grid.height - 1)):
                if grid.get_cell(i, j) == CellType.EMPTY:
                    grid.set_cell(i, j, CellType.FLOOR)

    def _connect_zones(self, grid: BaseGrid, zone_plan: List[ZoneRequirements]):
        """Connect zones with paths or bridges"""
        if len(zone_plan) < 2:
            return

        # Simple connection - create paths between zone centers
        for i in range(len(zone_plan) - 1):
            zone1 = self._get_zone_for_requirement(zone_plan[i])
            zone2 = self._get_zone_for_requirement(zone_plan[i + 1])

            if zone1 and zone2:
                x1, y1, w1, h1 = zone1.bounds
                x2, y2, w2, h2 = zone2.bounds

                # Connect centers
                start = (x1 + w1 // 2, y1 + h1 // 2)
                end = (x2 + w2 // 2, y2 + h2 // 2)

                self._create_path(grid, start, end)

    def _get_zone_for_requirement(self, req: ZoneRequirements) -> Optional[TerrainZone]:
        """Get the terrain zone for a requirement"""
        for zone, zone_req in self.zone_assignments.items():
            if zone_req == req:
                return zone
        return None

    def _create_path(
        self, grid: BaseGrid, start: Tuple[int, int], end: Tuple[int, int]
    ):
        """Create a path between two points"""
        x1, y1 = start
        x2, y2 = end

        # Simple L-shaped path
        # Horizontal first
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 <= x < grid.width and 0 <= y1 < grid.height:
                if grid.get_cell(x, y1) == CellType.EMPTY:
                    grid.set_cell(x, y1, CellType.FLOOR)

        # Then vertical
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 <= x2 < grid.width and 0 <= y < grid.height:
                if grid.get_cell(x2, y) == CellType.EMPTY:
                    grid.set_cell(x2, y, CellType.FLOOR)

    def _add_outer_defenses(self, grid: BaseGrid, num_layers: int):
        """Add multiple defensive layers"""
        # Add concentric defensive walls
        for layer in range(1, num_layers):
            offset = layer * 10  # Space between defensive layers

            # Top and bottom walls
            for x in range(offset, grid.width - offset):
                if 0 <= offset < grid.height:
                    grid.set_cell(x, offset, CellType.WALL)
                if 0 <= grid.height - offset - 1 < grid.height:
                    grid.set_cell(x, grid.height - offset - 1, CellType.WALL)

            # Left and right walls
            for y in range(offset, grid.height - offset):
                if 0 <= offset < grid.width:
                    grid.set_cell(offset, y, CellType.WALL)
                if 0 <= grid.width - offset - 1 < grid.width:
                    grid.set_cell(grid.width - offset - 1, y, CellType.WALL)

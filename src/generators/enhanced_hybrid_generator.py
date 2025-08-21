"""
Enhanced hybrid generator that can use complete prefabs, parts of prefabs, or just decorative elements.
Provides more organic and varied generation by mixing prefab concepts at different scales.
"""

import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
from src.generators.alpha_prefab_parser import AlphaPrefabLayout
from src.generators.wfc_generator import TileType


class PrefabUsageMode(Enum):
    """How to use a prefab in generation"""

    COMPLETE = "complete"  # Use entire prefab as-is
    PARTIAL = "partial"  # Use part of the prefab (e.g., just one room)
    DECORATIVE = "decorative"  # Use only decorative elements
    CONCEPTUAL = "conceptual"  # Use layout concept but different materials


@dataclass
class PrefabFragment:
    """A fragment or concept extracted from a prefab"""

    source_prefab: str
    fragment_type: str  # "room", "decoration", "furniture_arrangement", "wall_pattern"
    layout: np.ndarray
    metadata: dict


class EnhancedHybridGenerator(HybridPrefabGenerator):
    """
    Enhanced generator that can use prefabs at multiple levels

    CLEAN SLATE APPROACH:
    - ALL existing structures are considered demolishable
    - Generator designs from scratch, ignoring current buildings
    - Only terrain features (water, mountains) are respected as constraints
    - Existing buildings do NOT influence or block generation plans
    """

    def __init__(
        self,
        width: int,
        height: int,
        alpha_prefabs_path: Path | None = None,
        learned_patterns_file: Path | None = None,
        game_state=None,
    ):
        super().__init__(width, height, alpha_prefabs_path, learned_patterns_file)
        self.game_state = (
            game_state  # Used ONLY for terrain analysis, NOT for preserving buildings
        )

        self.prefab_fragments: dict[str, list[PrefabFragment]] = {}
        self.decorative_patterns: dict[str, np.ndarray] = {}

        if alpha_prefabs_path:
            self._extract_prefab_components()

    def _extract_prefab_components(self):
        """Extract reusable components from prefabs"""
        print("Extracting prefab components...")

        for category, prefabs in self.prefab_library.items():
            for prefab in prefabs[:10]:  # Analyze first 10 of each category
                # Extract individual rooms
                rooms = self._extract_rooms(prefab)
                for room in rooms:
                    if category not in self.prefab_fragments:
                        self.prefab_fragments[category] = []
                    self.prefab_fragments[category].append(room)

                # Extract decorative patterns
                patterns = self._extract_decorative_patterns(prefab)
                for pattern_name, pattern in patterns.items():
                    self.decorative_patterns[f"{prefab.def_name}_{pattern_name}"] = (
                        pattern
                    )

                # Extract furniture arrangements
                furniture = self._extract_furniture_arrangements(prefab)
                for arrangement in furniture:
                    key = f"{category}_furniture"
                    if key not in self.prefab_fragments:
                        self.prefab_fragments[key] = []
                    self.prefab_fragments[key].append(arrangement)

    def _extract_rooms(self, prefab: AlphaPrefabLayout) -> list[PrefabFragment]:
        """Extract individual rooms from a prefab"""
        rooms = []

        # Simple room detection: find enclosed areas
        layout = self._layout_to_array(prefab)
        visited = np.zeros_like(layout, dtype=bool)

        for y in range(1, layout.shape[0] - 1):
            for x in range(1, layout.shape[1] - 1):
                if not visited[y, x] and layout[y, x] != 1:  # Not wall and not visited
                    # Flood fill to find room
                    room_tiles = self._flood_fill_room(layout, x, y, visited)
                    if 4 <= len(room_tiles) <= 50:  # Reasonable room size
                        # Extract room bounds
                        min_x = min(t[0] for t in room_tiles)
                        max_x = max(t[0] for t in room_tiles)
                        min_y = min(t[1] for t in room_tiles)
                        max_y = max(t[1] for t in room_tiles)

                        # Include walls
                        room_layout = layout[
                            min_y - 1 : max_y + 2, min_x - 1 : max_x + 2
                        ].copy()

                        rooms.append(
                            PrefabFragment(
                                source_prefab=prefab.def_name,
                                fragment_type="room",
                                layout=room_layout,
                                metadata={
                                    "size": len(room_tiles),
                                    "aspect_ratio": (max_x - min_x) / (max_y - min_y)
                                    if max_y != min_y
                                    else 1,
                                },
                            )
                        )

        return rooms

    def _extract_decorative_patterns(
        self, prefab: AlphaPrefabLayout
    ) -> dict[str, np.ndarray]:
        """Extract decorative patterns like floor patterns, wall decorations"""
        patterns = {}

        # Extract floor patterns if terrain grid exists
        if prefab.terrain_grid:
            # Look for interesting floor patterns (not just uniform)
            terrain_array = self._terrain_to_array(prefab.terrain_grid)
            unique_tiles = np.unique(terrain_array)
            if len(unique_tiles) > 2:  # Has pattern variety
                patterns["floor_pattern"] = terrain_array

        # Extract corner decorations (e.g., plants, lamps in corners)
        layout = self._layout_to_array(prefab)
        corner_pattern = self._find_corner_decorations(layout)
        if corner_pattern is not None:
            patterns["corner_decoration"] = corner_pattern

        return patterns

    def _extract_furniture_arrangements(
        self, prefab: AlphaPrefabLayout
    ) -> list[PrefabFragment]:
        """Extract furniture placement patterns"""
        arrangements = []
        layout = self._layout_to_array(prefab)

        # Look for furniture clusters (non-wall, non-empty tiles grouped together)
        furniture_mask = layout > 2  # Furniture types are > 2

        if np.any(furniture_mask):
            # Find connected furniture groups
            visited = np.zeros_like(furniture_mask, dtype=bool)
            for y in range(layout.shape[0]):
                for x in range(layout.shape[1]):
                    if furniture_mask[y, x] and not visited[y, x]:
                        cluster = self._find_furniture_cluster(layout, x, y, visited)
                        if 2 <= len(cluster) <= 10:  # Reasonable cluster size
                            min_x = min(t[0] for t in cluster)
                            max_x = max(t[0] for t in cluster)
                            min_y = min(t[1] for t in cluster)
                            max_y = max(t[1] for t in cluster)

                            arrangement_layout = layout[
                                min_y : max_y + 1, min_x : max_x + 1
                            ].copy()

                            arrangements.append(
                                PrefabFragment(
                                    source_prefab=prefab.def_name,
                                    fragment_type="furniture_arrangement",
                                    layout=arrangement_layout,
                                    metadata={"items": len(cluster)},
                                )
                            )

        return arrangements

    def generate_enhanced(
        self,
        buildable_mask: np.ndarray | None = None,
        usage_modes: list[PrefabUsageMode] = None,
        prefab_density: float = 0.3,
        decoration_density: float = 0.2,
    ) -> np.ndarray:
        """
        Generate base using prefabs at multiple scales.

        Args:
            buildable_mask: Boolean array of buildable areas
            usage_modes: How to use prefabs (complete, partial, decorative, conceptual)
            prefab_density: How much of the base should use prefab elements (0-1)
            decoration_density: How much decoration to add (0-1)

        Returns:
            Generated grid
        """
        # Reset grid
        self.reset()

        if buildable_mask is not None:
            self._apply_buildable_mask(buildable_mask)

        if usage_modes is None:
            usage_modes = [
                PrefabUsageMode.COMPLETE,
                PrefabUsageMode.PARTIAL,
                PrefabUsageMode.DECORATIVE,
            ]

        # Phase 1: Place complete prefabs as major anchors (less dense)
        if PrefabUsageMode.COMPLETE in usage_modes:
            num_complete = int(3 * prefab_density)
            self._place_complete_prefabs(num_complete)

        # Phase 2: Place partial prefabs (individual rooms)
        if PrefabUsageMode.PARTIAL in usage_modes:
            num_partial = int(5 * prefab_density)
            self._place_partial_prefabs(num_partial)

        # Phase 3: Fill with WFC
        self._fill_with_wfc()

        # Phase 4: Add decorative elements
        if PrefabUsageMode.DECORATIVE in usage_modes:
            self._add_decorative_elements(decoration_density)

        # Phase 5: Apply conceptual patterns (wall styles, etc.)
        if PrefabUsageMode.CONCEPTUAL in usage_modes:
            self._apply_conceptual_patterns()

        # Connect everything
        self._connect_all_rooms()

        return self.grid

    def _place_complete_prefabs(self, count: int):
        """Place complete prefabs as anchors"""
        placed = 0
        categories = list(self.prefab_library.keys())
        random.shuffle(categories)

        for category in categories:
            if placed >= count:
                break
            prefabs = self.prefab_library[category]
            if prefabs:
                prefab = random.choice(prefabs)
                if self._place_prefab(prefab, category):
                    placed += 1
                    print(f"  Placed complete: {prefab.def_name}")

    def _place_partial_prefabs(self, count: int):
        """Place individual rooms extracted from prefabs"""
        placed = 0

        for category, fragments in self.prefab_fragments.items():
            if placed >= count:
                break
            if "room" in category or fragments:
                for fragment in random.sample(fragments, min(2, len(fragments))):
                    if fragment.fragment_type == "room":
                        if self._place_fragment(fragment):
                            placed += 1
                            print(f"  Placed room from: {fragment.source_prefab}")
                            if placed >= count:
                                break

    def _place_fragment(self, fragment: PrefabFragment) -> bool:
        """Place a prefab fragment on the grid"""
        height, width = fragment.layout.shape

        # Find valid position
        for _ in range(20):
            x = random.randint(1, self.width - width - 1)
            y = random.randint(1, self.height - height - 1)

            if self._is_valid_fragment_position(x, y, width, height):
                # Place the fragment
                for dy in range(height):
                    for dx in range(width):
                        if fragment.layout[dy, dx] > 0:  # Not empty
                            self.grid[y + dy, x + dx] = fragment.layout[dy, dx]
                            self.collapsed[y + dy, x + dx] = True
                return True

        return False

    def _is_valid_fragment_position(
        self, x: int, y: int, width: int, height: int
    ) -> bool:
        """Check if fragment can be placed"""
        if x < 0 or y < 0 or x + width >= self.width or y + height >= self.height:
            return False

        # Check overlaps (allow some overlap for organic feel)
        overlap_count = 0
        for dy in range(height):
            for dx in range(width):
                if self.collapsed[y + dy, x + dx]:
                    overlap_count += 1

        return overlap_count < (width * height * 0.2)  # Allow 20% overlap

    def _add_decorative_elements(self, density: float):
        """Add decorative patterns and furniture arrangements"""
        # Add furniture arrangements
        furniture_keys = [k for k in self.prefab_fragments.keys() if "furniture" in k]
        if furniture_keys:
            num_furniture = int(10 * density)
            for _ in range(num_furniture):
                key = random.choice(furniture_keys)
                if self.prefab_fragments[key]:
                    fragment = random.choice(self.prefab_fragments[key])
                    self._place_fragment_as_decoration(fragment)

        # Add corner decorations
        if self.decorative_patterns:
            self._add_corner_decorations(density)

    def _place_fragment_as_decoration(self, fragment: PrefabFragment):
        """Place fragment as decoration in existing rooms"""
        # Find empty areas in rooms
        for _ in range(10):
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)

            # Check if in a room (surrounded by non-empty)
            if self.grid[y, x] > TileType.WALL.value:  # In a room
                height, width = fragment.layout.shape
                if x + width < self.width and y + height < self.height:
                    # Place only furniture items (not walls)
                    for dy in range(height):
                        for dx in range(width):
                            if fragment.layout[dy, dx] > 2:  # Furniture
                                if self.grid[y + dy, x + dx] == TileType.EMPTY.value:
                                    self.grid[y + dy, x + dx] = fragment.layout[dy, dx]
                    return

    def _add_corner_decorations(self, density: float):
        """Add decorations to room corners"""
        # Find room corners
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if random.random() < density:
                    if self._is_room_corner(x, y):
                        # Add a decorative element
                        self.grid[y, x] = TileType.RECREATION.value  # Use as decoration

    def _is_room_corner(self, x: int, y: int) -> bool:
        """Check if position is a room corner"""
        if self.grid[y, x] != TileType.EMPTY.value:
            return False

        # Check if two adjacent tiles are walls
        walls = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                if self.grid[y + dy, x + dx] == TileType.WALL.value:
                    walls += 1

        return walls == 2

    def _apply_conceptual_patterns(self):
        """Apply design concepts from prefabs without copying directly"""
        # Example: Apply wall thickness patterns
        # Some prefabs use double walls, others single
        if random.random() < 0.3:  # 30% chance of double walls
            self._thicken_walls()

        # Apply symmetry patterns learned from prefabs
        if random.random() < 0.4:  # 40% chance of symmetry
            self._apply_symmetry()

    def _thicken_walls(self):
        """Make walls thicker like some defensive prefabs"""
        original_grid = self.grid.copy()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if original_grid[y, x] == TileType.WALL.value:
                    # Check if external wall (next to empty)
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if original_grid[ny, nx] == TileType.EMPTY.value:
                                # Thicken this wall
                                self.grid[ny, nx] = TileType.WALL.value
                                break

    def _apply_symmetry(self):
        """Apply symmetrical patterns to parts of the base"""
        # Find the center
        center_x = self.width // 2

        # Mirror a portion of the base
        for y in range(self.height // 3, 2 * self.height // 3):
            for x in range(center_x):
                mirror_x = 2 * center_x - x - 1
                if mirror_x < self.width:
                    if self.grid[y, x] in [
                        TileType.BEDROOM.value,
                        TileType.STORAGE.value,
                    ]:
                        self.grid[y, mirror_x] = self.grid[y, x]

    def _connect_all_rooms(self):
        """Connect all rooms and prefabs"""
        # Use parent method plus additional connections
        super()._connect_prefabs()

        # Add more organic connections
        self._add_organic_corridors()

    def _add_organic_corridors(self):
        """Add more natural-looking corridors"""
        # Find disconnected areas and connect them
        for _ in range(5):
            x1 = random.randint(self.width // 4, 3 * self.width // 4)
            y1 = random.randint(self.height // 4, 3 * self.height // 4)
            x2 = random.randint(self.width // 4, 3 * self.width // 4)
            y2 = random.randint(self.height // 4, 3 * self.height // 4)

            # Create curved corridor
            self._create_curved_corridor(x1, y1, x2, y2)

    def _create_curved_corridor(self, x1: int, y1: int, x2: int, y2: int):
        """Create a curved corridor between two points"""
        # Simple curve: go partially toward target, then adjust
        mid_x = (x1 + x2) // 2 + random.randint(-3, 3)
        mid_y = (y1 + y2) // 2 + random.randint(-3, 3)

        # Connect three points
        for x, y in self._bresenham_line(x1, y1, mid_x, mid_y):
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[y, x] == TileType.EMPTY.value:
                    self.grid[y, x] = TileType.CORRIDOR.value

        for x, y in self._bresenham_line(mid_x, mid_y, x2, y2):
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[y, x] == TileType.EMPTY.value:
                    self.grid[y, x] = TileType.CORRIDOR.value

    def _bresenham_line(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> list[tuple[int, int]]:
        """Generate points along a line using Bresenham's algorithm"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    # Helper methods

    def _layout_to_array(self, prefab: AlphaPrefabLayout) -> np.ndarray:
        """Convert prefab layout to numpy array"""
        array = np.zeros((prefab.height, prefab.width), dtype=int)
        for y, row in enumerate(prefab.layout_grid):
            for x, item in enumerate(row):
                array[y, x] = self._map_item_to_tile(item).value
        return array

    def _terrain_to_array(self, terrain_grid: list[list[str]]) -> np.ndarray:
        """Convert terrain grid to numpy array"""
        height = len(terrain_grid)
        width = len(terrain_grid[0]) if terrain_grid else 0
        array = np.zeros((height, width), dtype=int)

        terrain_map = {
            ".": 0,
            "WoodPlankFloor": 1,
            "StrawMatting": 2,
            "CarpetPurpleDeep": 3,
            "TileLimestone": 4,
        }

        for y, row in enumerate(terrain_grid):
            for x, terrain in enumerate(row):
                array[y, x] = terrain_map.get(terrain, 0)

        return array

    def _flood_fill_room(
        self, layout: np.ndarray, start_x: int, start_y: int, visited: np.ndarray
    ) -> list[tuple[int, int]]:
        """Flood fill to find room tiles"""
        room_tiles = []
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack.pop()
            if visited[y, x] or layout[y, x] == 1:  # Already visited or is wall
                continue

            visited[y, x] = True
            room_tiles.append((x, y))

            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < layout.shape[1] and 0 <= ny < layout.shape[0]:
                    if not visited[ny, nx] and layout[ny, nx] != 1:
                        stack.append((nx, ny))

        return room_tiles

    def _find_corner_decorations(self, layout: np.ndarray) -> np.ndarray | None:
        """Find decorative patterns in corners"""
        # Look for non-wall, non-empty items in corners
        corner_items = []
        h, w = layout.shape

        corners = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
        for y, x in corners:
            if 0 <= y < h and 0 <= x < w:
                if layout[y, x] > 2:  # Decoration/furniture
                    corner_items.append(layout[y, x])

        if corner_items:
            return np.array(corner_items)
        return None

    def _find_furniture_cluster(
        self, layout: np.ndarray, start_x: int, start_y: int, visited: np.ndarray
    ) -> list[tuple[int, int]]:
        """Find connected furniture items"""
        cluster = []
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack.pop()
            if visited[y, x] or layout[y, x] <= 2:  # Already visited or not furniture
                continue

            visited[y, x] = True
            cluster.append((x, y))

            # Check neighbors (including diagonals for furniture)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < layout.shape[1] and 0 <= ny < layout.shape[0]:
                        if not visited[ny, nx] and layout[ny, nx] > 2:
                            stack.append((nx, ny))

        return cluster

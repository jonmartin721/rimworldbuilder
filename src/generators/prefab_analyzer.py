"""
Prefab Analyzer - Learns from existing RimWorld prefab designs to generate better bases.
Analyzes patterns in pre-made designs from mods like Prefabs to understand:
- Room size distributions
- Adjacency patterns
- Traffic flow
- Defensive layouts
- Efficiency patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PrefabDesign:
    """Represents a prefab design from a mod"""

    name: str
    width: int
    height: int
    layout: np.ndarray  # 2D array of tile types
    metadata: Dict  # Additional info (author, purpose, etc.)

    def get_rooms(self) -> List["Room"]:
        """Extract individual rooms from the design"""
        # Use flood fill to identify connected components (rooms)
        visited = np.zeros_like(self.layout, dtype=bool)
        rooms = []

        for y in range(self.height):
            for x in range(self.width):
                if not visited[y, x] and self.layout[y, x] not in [
                    0,
                    1,
                ]:  # Not empty or wall
                    room = self._flood_fill_room(x, y, visited)
                    if room:
                        rooms.append(room)

        return rooms

    def _flood_fill_room(
        self, start_x: int, start_y: int, visited: np.ndarray
    ) -> Optional["Room"]:
        """Extract a room using flood fill"""
        room_tiles = []
        stack = [(start_x, start_y)]
        room_type = self.layout[start_y, start_x]

        while stack:
            x, y = stack.pop()

            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            if visited[y, x] or self.layout[y, x] != room_type:
                continue

            visited[y, x] = True
            room_tiles.append((x, y))

            # Add neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((x + dx, y + dy))

        if room_tiles:
            return Room(tiles=room_tiles, room_type=int(room_type))
        return None


@dataclass
class Room:
    """Represents a room extracted from a prefab"""

    tiles: List[Tuple[int, int]]
    room_type: int

    @property
    def area(self) -> int:
        return len(self.tiles)

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box (min_x, min_y, max_x, max_y)"""
        xs = [x for x, y in self.tiles]
        ys = [y for x, y in self.tiles]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio of bounding box"""
        min_x, min_y, max_x, max_y = self.bounds
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return width / height if height > 0 else 1.0


class PrefabAnalyzer:
    """Analyzes prefab designs to learn patterns"""

    def __init__(self):
        self.prefabs: List[PrefabDesign] = []
        self.patterns = {
            "room_sizes": defaultdict(list),  # room_type -> [sizes]
            "room_shapes": defaultdict(list),  # room_type -> [aspect_ratios]
            "adjacencies": defaultdict(
                lambda: defaultdict(int)
            ),  # room_type -> {adjacent_type: count}
            "door_positions": defaultdict(list),  # room_type -> [relative_positions]
            "traffic_patterns": [],  # Common corridor/path patterns
            "defensive_patterns": [],  # Wall and turret placement patterns
        }

    def load_prefab_from_xml(self, xml_path: Path) -> Optional[PrefabDesign]:
        """
        Load a prefab design from RimWorld mod XML format.
        This would parse the actual prefab mod files.
        """
        # TODO: Implement actual XML parsing for prefab mods
        # For now, return a mock prefab for testing
        return self._create_mock_prefab()

    def _create_mock_prefab(self) -> PrefabDesign:
        """Create a mock prefab for testing - represents a typical efficient base"""
        # Legend: 0=empty, 1=wall, 2=door, 3=bedroom, 4=kitchen, 5=dining,
        #         6=workshop, 7=storage, 8=rec, 9=hospital, 10=power

        layout = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 3, 3, 3, 1, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 7, 7, 7, 1],
                [1, 3, 3, 3, 2, 4, 4, 4, 2, 5, 5, 5, 5, 5, 5, 2, 7, 7, 7, 1],
                [1, 3, 3, 3, 1, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 7, 7, 7, 1],
                [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1],
                [1, 3, 3, 3, 1, 6, 6, 6, 6, 6, 1, 8, 8, 8, 1, 7, 7, 7, 7, 1],
                [1, 3, 3, 3, 2, 6, 6, 6, 6, 6, 2, 8, 8, 8, 2, 7, 7, 7, 7, 1],
                [1, 3, 3, 3, 1, 6, 6, 6, 6, 6, 1, 8, 8, 8, 1, 7, 7, 7, 7, 1],
                [1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1],
                [1, 3, 3, 3, 1, 9, 9, 9, 9, 1, 10, 10, 10, 1, 7, 7, 7, 7, 1],
                [1, 3, 3, 3, 2, 9, 9, 9, 9, 2, 10, 10, 10, 2, 7, 7, 7, 7, 1],
                [1, 3, 3, 3, 1, 9, 9, 9, 9, 1, 10, 10, 10, 1, 7, 7, 7, 7, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        return PrefabDesign(
            name="efficient_base_template",
            width=20,
            height=13,
            layout=layout,
            metadata={"author": "analyzer", "purpose": "general", "efficiency": "high"},
        )

    def analyze_prefab(self, prefab: PrefabDesign):
        """Analyze a single prefab design to extract patterns"""
        self.prefabs.append(prefab)

        # Extract rooms
        rooms = prefab.get_rooms()

        # Analyze room sizes and shapes
        for room in rooms:
            self.patterns["room_sizes"][room.room_type].append(room.area)
            self.patterns["room_shapes"][room.room_type].append(room.aspect_ratio)

        # Analyze adjacencies
        self._analyze_adjacencies(prefab.layout)

        # Analyze door positions
        self._analyze_doors(prefab.layout)

        # Analyze traffic flow
        self._analyze_traffic(prefab.layout)

        logger.info(f"Analyzed prefab: {prefab.name}")

    def _analyze_adjacencies(self, layout: np.ndarray):
        """Analyze which room types are commonly adjacent"""
        height, width = layout.shape

        for y in range(height):
            for x in range(width):
                current = layout[y, x]
                if current <= 2:  # Skip walls, doors, empty
                    continue

                # Check neighbors
                for dx, dy in [
                    (0, 1),
                    (1, 0),
                ]:  # Only check right and down to avoid duplicates
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = layout[ny, nx]
                        if neighbor > 2:  # Another room type
                            self.patterns["adjacencies"][current][neighbor] += 1
                            self.patterns["adjacencies"][neighbor][current] += 1

    def _analyze_doors(self, layout: np.ndarray):
        """Analyze door placement patterns"""
        height, width = layout.shape

        for y in range(height):
            for x in range(width):
                if layout[y, x] == 2:  # Door
                    # Find adjacent room types
                    adjacent_rooms = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if layout[ny, nx] > 2:
                                adjacent_rooms.append(layout[ny, nx])

                    # Store door position relative to room
                    for room_type in adjacent_rooms:
                        self.patterns["door_positions"][room_type].append((x, y))

    def _analyze_traffic(self, layout: np.ndarray):
        """Analyze traffic flow patterns"""
        # Find main corridors (areas with high door connectivity)
        height, width = layout.shape
        traffic_heat = np.zeros_like(layout, dtype=float)

        # Door positions increase traffic heat
        for y in range(height):
            for x in range(width):
                if layout[y, x] == 2:  # Door
                    # Spread heat to nearby tiles
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                distance = abs(dy) + abs(dx)
                                if distance > 0:
                                    traffic_heat[ny, nx] += 1.0 / distance

        self.patterns["traffic_patterns"].append(traffic_heat)

    def get_learned_rules(self) -> Dict:
        """
        Extract learned rules from analyzed prefabs.
        These rules can be used to generate new designs.
        """
        rules = {}

        # Calculate average room sizes
        avg_sizes = {}
        for room_type, sizes in self.patterns["room_sizes"].items():
            if sizes:
                avg_sizes[room_type] = {
                    "mean": np.mean(sizes),
                    "std": np.std(sizes),
                    "min": min(sizes),
                    "max": max(sizes),
                }
        rules["room_sizes"] = avg_sizes

        # Calculate common adjacencies
        adjacency_rules = {}
        for room_type, neighbors in self.patterns["adjacencies"].items():
            if neighbors:
                total = sum(neighbors.values())
                adjacency_rules[room_type] = {
                    n: count / total for n, count in neighbors.items()
                }
        rules["adjacencies"] = adjacency_rules

        # Calculate average aspect ratios
        avg_shapes = {}
        for room_type, shapes in self.patterns["room_shapes"].items():
            if shapes:
                avg_shapes[room_type] = {"mean": np.mean(shapes), "std": np.std(shapes)}
        rules["room_shapes"] = avg_shapes

        # Traffic patterns - only average if all are same size
        if self.patterns["traffic_patterns"]:
            # Check if all traffic patterns have same shape
            shapes = [p.shape for p in self.patterns["traffic_patterns"]]
            if len(set(shapes)) == 1:  # All same shape
                avg_traffic = np.mean(self.patterns["traffic_patterns"], axis=0)
                rules["traffic_heat"] = avg_traffic.tolist()
            else:
                # Different sizes, store the first one as example
                rules["traffic_heat"] = self.patterns["traffic_patterns"][0].tolist()

        return rules

    def generate_from_learned_patterns(
        self, width: int, height: int, required_rooms: List[int]
    ) -> np.ndarray:
        """
        Generate a new base layout using learned patterns.
        This is a simplified version - a full implementation would use
        more sophisticated algorithms.
        """
        layout = np.zeros((height, width), dtype=int)
        rules = self.get_learned_rules()

        # Place rooms based on learned sizes and adjacencies
        placed_rooms = []

        for room_type in required_rooms:
            # Get expected size
            size_info = rules["room_sizes"].get(room_type, {"mean": 9, "std": 3})
            room_size = max(
                4, int(np.random.normal(size_info["mean"], size_info["std"]))
            )

            # Get expected shape
            shape_info = rules["room_shapes"].get(room_type, {"mean": 1.0, "std": 0.3})
            aspect_ratio = max(
                0.5, np.random.normal(shape_info["mean"], shape_info["std"])
            )

            # Calculate room dimensions
            room_height = int(np.sqrt(room_size / aspect_ratio))
            room_width = int(room_size / room_height)

            # Try to place room (simplified placement)
            placed = self._place_room_in_layout(
                layout,
                room_type,
                room_width,
                room_height,
                placed_rooms,
                rules.get("adjacencies", {}),
            )

            if placed:
                placed_rooms.append((room_type, placed))

        return layout

    def _place_room_in_layout(
        self,
        layout: np.ndarray,
        room_type: int,
        room_width: int,
        room_height: int,
        placed_rooms: List,
        adjacency_rules: Dict,
    ) -> Optional[Tuple[int, int]]:
        """Try to place a room in the layout based on learned rules"""
        height, width = layout.shape

        # Find best position based on adjacency preferences
        best_score = -1
        best_pos = None

        for y in range(height - room_height):
            for x in range(width - room_width):
                # Check if space is available
                if np.any(layout[y : y + room_height, x : x + room_width] != 0):
                    continue

                # Calculate adjacency score
                score = self._calculate_placement_score(
                    layout, x, y, room_width, room_height, room_type, adjacency_rules
                )

                if score > best_score:
                    best_score = score
                    best_pos = (x, y)

        if best_pos:
            x, y = best_pos
            # Place room
            layout[y : y + room_height, x : x + room_width] = room_type
            # Add walls
            layout[y, x : x + room_width] = 1
            layout[y + room_height - 1, x : x + room_width] = 1
            layout[y : y + room_height, x] = 1
            layout[y : y + room_height, x + room_width - 1] = 1
            # Add door
            layout[y + room_height // 2, x] = 2

            return best_pos

        return None

    def _calculate_placement_score(
        self,
        layout: np.ndarray,
        x: int,
        y: int,
        room_width: int,
        room_height: int,
        room_type: int,
        adjacency_rules: Dict,
    ) -> float:
        """Calculate how good a placement is based on learned adjacency rules"""
        score = 0.0
        height, width = layout.shape

        # Check what's adjacent to this position
        for dy in range(-1, room_height + 1):
            for dx in range(-1, room_width + 1):
                # Only check edges
                if 0 < dy < room_height - 1 and 0 < dx < room_width - 1:
                    continue

                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor = layout[ny, nx]
                    if neighbor > 2:  # Another room
                        # Check if this is a good neighbor
                        if room_type in adjacency_rules:
                            score += adjacency_rules[room_type].get(neighbor, 0)

        return score

    def save_learned_patterns(self, filepath: Path):
        """Save learned patterns to a file"""
        rules = self.get_learned_rules()

        # Convert numpy arrays and int64 to JSON-serializable types
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {
                    make_serializable(k): make_serializable(v) for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_rules = make_serializable(rules)

        with open(filepath, "w") as f:
            json.dump(serializable_rules, f, indent=2)

        logger.info(f"Saved learned patterns to {filepath}")

    def load_learned_patterns(self, filepath: Path):
        """Load previously learned patterns"""
        with open(filepath, "r") as f:
            rules = json.load(f)

        # Convert lists back to numpy arrays where needed
        if "traffic_heat" in rules:
            rules["traffic_heat"] = np.array(rules["traffic_heat"])

        # Update patterns
        # This is simplified - full implementation would properly merge patterns
        logger.info(f"Loaded learned patterns from {filepath}")

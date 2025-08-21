"""
Improved Wave Function Collapse generator using learned patterns from real prefabs.
Integrates room templates and adjacency rules from analyzed AlphaPrefabs data.
"""

import numpy as np
import random
import json
from pathlib import Path

from src.generators.wfc_generator import WFCGenerator, TileType
from src.generators.room_templates import RoomTemplateLibrary as RoomTemplates


class ImprovedWFCGenerator(WFCGenerator):
    """Enhanced WFC generator using learned patterns"""

    def __init__(
        self, width: int, height: int, learned_patterns_file: Path | None = None
    ):
        """
        Initialize improved generator with learned patterns.

        Args:
            width: Grid width
            height: Grid height
            learned_patterns_file: Path to JSON file with learned patterns
        """
        super().__init__(width, height)

        # Load learned patterns if provided
        self.learned_patterns = {}
        if learned_patterns_file and learned_patterns_file.exists():
            with open(learned_patterns_file) as f:
                self.learned_patterns = json.load(f)
            self._update_adjacency_from_patterns()

        # Room placement tracking
        self.room_placements = []
        self.room_templates = RoomTemplates()

    def _update_adjacency_from_patterns(self):
        """Update adjacency rules based on learned patterns"""
        if "adjacencies" not in self.learned_patterns:
            return

        adjacencies = self.learned_patterns["adjacencies"]

        # Map learned room types to our TileTypes
        type_mapping = {
            "3": TileType.BEDROOM,
            "4": TileType.KITCHEN,
            "5": TileType.RECREATION,
            "6": TileType.STORAGE,
            "7": TileType.WORKSHOP,
            "8": TileType.HOSPITAL,
            "9": TileType.RESEARCH,
            "10": TileType.POWER,
        }

        # Update adjacency rules based on learned preferences
        for room_type_str, neighbors in adjacencies.items():
            if room_type_str in type_mapping:
                tile_type = type_mapping[room_type_str]

                # Clear existing adjacencies for this type
                self.adjacency_rules[tile_type] = set()

                # Add learned adjacencies
                for neighbor_str, probability in neighbors.items():
                    if neighbor_str in type_mapping and probability > 0.1:
                        neighbor_type = type_mapping[neighbor_str]
                        self.adjacency_rules[tile_type].add(neighbor_type)

                # Always allow walls and corridors
                self.adjacency_rules[tile_type].add(TileType.WALL)
                self.adjacency_rules[tile_type].add(TileType.CORRIDOR)

    def reset(self):
        """Reset the grid to empty state"""
        self.grid = np.full(
            (self.height, self.width), TileType.EMPTY.value, dtype=object
        )
        self.collapsed = np.zeros((self.height, self.width), dtype=bool)

    def generate_with_templates(
        self,
        buildable_mask: np.ndarray | None = None,
        num_bedrooms: int = 4,
        num_workrooms: int = 2,
        include_kitchen: bool = True,
        include_rec: bool = True,
        include_storage: bool = True,
    ) -> np.ndarray:
        """
        Generate base layout using room templates and learned patterns.

        Args:
            buildable_mask: Boolean array indicating buildable areas
            num_bedrooms: Number of bedrooms to place
            num_workrooms: Number of workshop/production rooms
            include_kitchen: Whether to include kitchen/dining
            include_rec: Whether to include recreation room
            include_storage: Whether to include storage areas

        Returns:
            Generated grid
        """
        # Reset grid
        self.reset()

        # Apply buildable mask if provided
        if buildable_mask is not None:
            self._apply_buildable_mask(buildable_mask)

        # Plan room layout
        room_plan = self._plan_room_layout(
            num_bedrooms, num_workrooms, include_kitchen, include_rec, include_storage
        )

        # Place rooms according to plan
        for room_info in room_plan:
            self._place_room_template(room_info)

        # Connect rooms with corridors
        self._connect_rooms_smart()

        # Fill remaining space with walls
        self._fill_remaining_space()

        # Grid is now complete
        return self.grid

    def _plan_room_layout(
        self,
        num_bedrooms: int,
        num_workrooms: int,
        include_kitchen: bool,
        include_rec: bool,
        include_storage: bool,
    ) -> list[dict]:
        """Plan optimal room layout based on learned patterns"""
        room_plan = []

        # Determine room sizes from learned patterns
        room_sizes = self._get_learned_room_sizes()

        # Core rooms (prioritized placement)
        if include_kitchen:
            kitchen_size = room_sizes.get(TileType.KITCHEN, (5, 4))
            room_plan.append(
                {
                    "type": TileType.KITCHEN,
                    "size": kitchen_size,
                    "priority": 1,
                    "central": True,  # Kitchen should be central
                }
            )

        if include_storage:
            storage_size = room_sizes.get(TileType.STORAGE, (6, 4))
            room_plan.append(
                {
                    "type": TileType.STORAGE,
                    "size": storage_size,
                    "priority": 2,
                    "central": True,
                }
            )

        # Bedrooms (placed around edges)
        bedroom_size = room_sizes.get(TileType.BEDROOM, (4, 3))
        for i in range(num_bedrooms):
            room_plan.append(
                {
                    "type": TileType.BEDROOM,
                    "size": bedroom_size,
                    "priority": 3,
                    "central": False,
                }
            )

        # Work rooms
        workshop_size = room_sizes.get(TileType.WORKSHOP, (5, 5))
        for i in range(num_workrooms):
            room_plan.append(
                {
                    "type": TileType.WORKSHOP,
                    "size": workshop_size,
                    "priority": 4,
                    "central": False,
                }
            )

        # Recreation (optional, lower priority)
        if include_rec:
            rec_size = room_sizes.get(TileType.RECREATION, (6, 5))
            room_plan.append(
                {
                    "type": TileType.RECREATION,
                    "size": rec_size,
                    "priority": 5,
                    "central": False,
                }
            )

        # Sort by priority
        room_plan.sort(key=lambda x: x["priority"])

        return room_plan

    def _get_learned_room_sizes(self) -> dict[TileType, tuple[int, int]]:
        """Get optimal room sizes from learned patterns"""
        sizes = {}

        if "room_sizes" not in self.learned_patterns:
            # Default sizes
            return {
                TileType.BEDROOM: (4, 3),
                TileType.KITCHEN: (5, 4),
                TileType.STORAGE: (6, 4),
                TileType.WORKSHOP: (5, 5),
                TileType.RECREATION: (6, 5),
                TileType.HOSPITAL: (4, 4),
                TileType.RESEARCH: (4, 3),
                TileType.POWER: (3, 3),
            }

        # Map learned sizes to our tile types
        type_mapping = {
            3: TileType.BEDROOM,
            4: TileType.KITCHEN,
            5: TileType.RECREATION,
            6: TileType.STORAGE,
            7: TileType.WORKSHOP,
            8: TileType.HOSPITAL,
            9: TileType.RESEARCH,
            10: TileType.POWER,
        }

        room_sizes = self.learned_patterns["room_sizes"]
        for room_type_key, stats in room_sizes.items():
            try:
                room_type_int = int(room_type_key)
                if room_type_int in type_mapping:
                    tile_type = type_mapping[room_type_int]
                    # Use average size to determine dimensions
                    avg_size = stats.get("mean", 12)
                    # Approximate as square-ish room
                    side = int(np.sqrt(avg_size))
                    sizes[tile_type] = (side + 1, side)
            except (ValueError, AttributeError):
                continue

        return sizes

    def _place_room_template(self, room_info: dict) -> bool:
        """Place a room template on the grid"""
        room_type = room_info["type"]
        width, height = room_info["size"]
        is_central = room_info.get("central", False)

        # Find valid placement position
        position = self._find_room_placement(width, height, is_central)
        if position is None:
            return False

        x, y = position

        # Place room walls
        for dy in range(height):
            for dx in range(width):
                px, py = x + dx, y + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    # Place walls on edges
                    if dx == 0 or dx == width - 1 or dy == 0 or dy == height - 1:
                        self.grid[py, px] = TileType.WALL.value
                        self.collapsed[py, px] = True
                    else:
                        # Interior of room
                        self.grid[py, px] = room_type.value
                        self.collapsed[py, px] = True

        # Add door
        self._add_room_door(x, y, width, height)

        # Track room placement
        self.room_placements.append(
            {"type": room_type, "position": (x, y), "size": (width, height)}
        )

        return True

    def _find_room_placement(
        self, width: int, height: int, is_central: bool
    ) -> tuple[int, int] | None:
        """Find valid position for room placement"""
        # Central rooms: place near center
        # Edge rooms: place along edges

        if is_central:
            # Try positions near center
            center_x = self.width // 2
            center_y = self.height // 2
            search_radius = min(self.width, self.height) // 4

            for r in range(0, search_radius, 2):
                for angle in range(0, 360, 45):
                    dx = int(r * np.cos(np.radians(angle)))
                    dy = int(r * np.sin(np.radians(angle)))
                    x = center_x + dx - width // 2
                    y = center_y + dy - height // 2

                    if self._is_valid_room_position(x, y, width, height):
                        return (x, y)
        else:
            # Try edge positions
            positions = []

            # Top edge
            for x in range(1, self.width - width - 1, 3):
                if self._is_valid_room_position(x, 1, width, height):
                    positions.append((x, 1))

            # Bottom edge
            for x in range(1, self.width - width - 1, 3):
                y = self.height - height - 1
                if self._is_valid_room_position(x, y, width, height):
                    positions.append((x, y))

            # Left edge
            for y in range(1, self.height - height - 1, 3):
                if self._is_valid_room_position(1, y, width, height):
                    positions.append((1, y))

            # Right edge
            for y in range(1, self.height - height - 1, 3):
                x = self.width - width - 1
                if self._is_valid_room_position(x, y, width, height):
                    positions.append((x, y))

            if positions:
                return random.choice(positions)

        # Fallback: any valid position
        for _ in range(100):
            x = random.randint(1, self.width - width - 1)
            y = random.randint(1, self.height - height - 1)
            if self._is_valid_room_position(x, y, width, height):
                return (x, y)

        return None

    def _is_valid_room_position(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if room can be placed at position"""
        # Check bounds
        if x < 0 or y < 0 or x + width >= self.width or y + height >= self.height:
            return False

        # Check for overlaps (with 1 tile buffer)
        for dy in range(-1, height + 1):
            for dx in range(-1, width + 1):
                px, py = x + dx, y + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    if (
                        self.collapsed[py, px]
                        and self.grid[py, px] != TileType.EMPTY.value
                    ):
                        return False

        return True

    def _add_room_door(self, x: int, y: int, width: int, height: int):
        """Add door to room"""
        # Choose random wall for door
        walls = []

        # Top wall
        for dx in range(1, width - 1):
            walls.append((x + dx, y, "horizontal"))

        # Bottom wall
        for dx in range(1, width - 1):
            walls.append((x + dx, y + height - 1, "horizontal"))

        # Left wall
        for dy in range(1, height - 1):
            walls.append((x, y + dy, "vertical"))

        # Right wall
        for dy in range(1, height - 1):
            walls.append((x + width - 1, y + dy, "vertical"))

        if walls:
            door_x, door_y, orientation = random.choice(walls)
            if 0 <= door_x < self.width and 0 <= door_y < self.height:
                self.grid[door_y, door_x] = TileType.DOOR.value
                self.collapsed[door_y, door_x] = True

    def _connect_rooms_smart(self):
        """Connect rooms with corridors using learned patterns"""
        # Create corridors between adjacent rooms
        for i, room1 in enumerate(self.room_placements):
            for room2 in self.room_placements[i + 1 :]:
                if self._should_connect_rooms(room1, room2):
                    self._create_corridor(room1, room2)

    def _should_connect_rooms(self, room1: dict, room2: dict) -> bool:
        """Determine if two rooms should be connected based on learned patterns"""
        if "adjacencies" not in self.learned_patterns:
            # Default: connect nearby rooms
            dist = self._room_distance(room1, room2)
            return dist < 15

        # Use learned adjacency preferences
        type1 = str(room1["type"].value)
        type2 = str(room2["type"].value)

        adjacencies = self.learned_patterns.get("adjacencies", {})
        if type1 in adjacencies:
            neighbors = adjacencies[type1]
            if type2 in neighbors:
                probability = neighbors[type2]
                return random.random() < probability

        # Default: connect if close
        dist = self._room_distance(room1, room2)
        return dist < 10

    def _room_distance(self, room1: dict, room2: dict) -> float:
        """Calculate distance between room centers"""
        x1 = room1["position"][0] + room1["size"][0] // 2
        y1 = room1["position"][1] + room1["size"][1] // 2
        x2 = room2["position"][0] + room2["size"][0] // 2
        y2 = room2["position"][1] + room2["size"][1] // 2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _create_corridor(self, room1: dict, room2: dict):
        """Create corridor between two rooms"""
        # Get room centers
        x1 = room1["position"][0] + room1["size"][0] // 2
        y1 = room1["position"][1] + room1["size"][1] // 2
        x2 = room2["position"][0] + room2["size"][0] // 2
        y2 = room2["position"][1] + room2["size"][1] // 2

        # Create L-shaped corridor
        # First go horizontal, then vertical
        if random.random() < 0.5:
            # Horizontal first
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < self.width and 0 <= y1 < self.height:
                    if not self.collapsed[y1, x]:
                        self.grid[y1, x] = TileType.CORRIDOR.value
                        self.collapsed[y1, x] = True

            # Then vertical
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x2 < self.width and 0 <= y < self.height:
                    if not self.collapsed[y, x2]:
                        self.grid[y, x2] = TileType.CORRIDOR.value
                        self.collapsed[y, x2] = True
        else:
            # Vertical first
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x1 < self.width and 0 <= y < self.height:
                    if not self.collapsed[y, x1]:
                        self.grid[y, x1] = TileType.CORRIDOR.value
                        self.collapsed[y, x1] = True

            # Then horizontal
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < self.width and 0 <= y2 < self.height:
                    if not self.collapsed[y2, x]:
                        self.grid[y2, x] = TileType.CORRIDOR.value
                        self.collapsed[y2, x] = True

    def _fill_remaining_space(self):
        """Fill remaining empty space appropriately"""
        for y in range(self.height):
            for x in range(self.width):
                if not self.collapsed[y, x]:
                    # Check neighbors manually
                    neighbor_coords = [
                        (x, y - 1),  # north
                        (x, y + 1),  # south
                        (x + 1, y),  # east
                        (x - 1, y),  # west
                    ]

                    # Get valid neighbor values
                    neighbor_types = []
                    for nx, ny in neighbor_coords:
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.collapsed[ny, nx]:
                                neighbor_types.append(self.grid[ny, nx])

                    # Place wall if next to rooms
                    if neighbor_types:
                        # Check if any neighbor is a room type (not empty/wall/corridor)
                        room_types = [
                            TileType.BEDROOM.value,
                            TileType.KITCHEN.value,
                            TileType.STORAGE.value,
                            TileType.WORKSHOP.value,
                            TileType.RECREATION.value,
                            TileType.HOSPITAL.value,
                        ]
                        if any(t in room_types for t in neighbor_types):
                            self.grid[y, x] = TileType.WALL.value
                        else:
                            self.grid[y, x] = TileType.EMPTY.value
                    else:
                        self.grid[y, x] = TileType.EMPTY.value
                    self.collapsed[y, x] = True

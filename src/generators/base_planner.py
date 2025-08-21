"""
Base planner that uses room templates to generate optimized RimWorld bases.
Specifically designed to work with bridge areas and other buildable zones.
"""

import random
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging

from src.models.game_entities import BuildingType
from src.generators.room_templates import RoomTemplateLibrary, RoomTemplate, RoomType

logger = logging.getLogger(__name__)


@dataclass
class PlacedRoom:
    """A room that has been placed in the base layout"""

    template: RoomTemplate
    x: int  # Top-left corner position
    y: int
    rotated: bool = False

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get (x_min, y_min, x_max, y_max) bounds"""
        return (
            self.x,
            self.y,
            self.x + self.template.width - 1,
            self.y + self.template.height - 1,
        )

    def overlaps_with(self, other: "PlacedRoom") -> bool:
        """Check if this room overlaps with another"""
        x1_min, y1_min, x1_max, y1_max = self.get_bounds()
        x2_min, y2_min, x2_max, y2_max = other.get_bounds()

        return not (
            x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min
        )


class BasePlanner:
    """Plans and generates RimWorld base layouts using room templates"""

    def __init__(
        self,
        width: int,
        height: int,
        buildable_positions: Optional[Set[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.buildable_positions = buildable_positions or {
            (x, y) for x in range(width) for y in range(height)
        }
        self.random = random.Random(seed)

        self.room_library = RoomTemplateLibrary()
        self.placed_rooms: List[PlacedRoom] = []
        self.grid = np.zeros(
            (height, width), dtype=int
        )  # 0=empty, 1=wall, 2=door, 3=room interior

        # Mark buildable areas
        self.buildable_grid = np.zeros((height, width), dtype=bool)
        for x, y in self.buildable_positions:
            if 0 <= x < width and 0 <= y < height:
                self.buildable_grid[y, x] = True

    def plan_base(self, required_rooms: Optional[List[RoomType]] = None) -> bool:
        """
        Plan a complete base layout.

        Args:
            required_rooms: List of room types that must be included

        Returns:
            True if successful, False if couldn't place all required rooms
        """
        # Default room requirements for a functional base
        if required_rooms is None:
            required_rooms = [
                RoomType.BEDROOM_SMALL,
                RoomType.BEDROOM_SMALL,
                RoomType.BEDROOM_SMALL,
                RoomType.KITCHEN,
                RoomType.DINING_HALL,
                RoomType.STORAGE_LARGE,
                RoomType.WORKSHOP,
                RoomType.POWER_ROOM,
                RoomType.HOSPITAL,
            ]

        # Get templates for required rooms
        templates_to_place = []
        for room_type in required_rooms:
            template = self.room_library.get_template(room_type)
            if template:
                templates_to_place.append(template)

        # Sort by priority and size (place important/large rooms first)
        templates_to_place.sort(
            key=lambda t: (t.priority, t.width * t.height), reverse=True
        )

        # Try to place each room
        for template in templates_to_place:
            placed = self._place_room(template)
            if not placed:
                logger.warning(f"Could not place {template.room_type.value}")

        # Connect rooms with corridors
        self._add_corridors()

        # Fill remaining buildable space with storage or misc rooms
        self._fill_remaining_space()

        return len(self.placed_rooms) > 0

    def _place_room(self, template: RoomTemplate, max_attempts: int = 100) -> bool:
        """Try to place a room template on the grid"""

        for attempt in range(max_attempts):
            # Try both orientations
            for rotated in [False, True]:
                current_template = template.rotate_90() if rotated else template

                # Find valid positions
                valid_positions = self._find_valid_positions(current_template)

                if valid_positions:
                    # Prefer positions near existing rooms (for connectivity)
                    if self.placed_rooms:
                        valid_positions.sort(
                            key=lambda p: min(
                                self._distance_to_room(p, room)
                                for room in self.placed_rooms
                            )
                        )

                    # Try to place at best position
                    for x, y in valid_positions[:10]:  # Try top 10 positions
                        if self._can_place_room_at(current_template, x, y):
                            self._place_room_at(current_template, x, y, rotated)
                            return True

        return False

    def _find_valid_positions(self, template: RoomTemplate) -> List[Tuple[int, int]]:
        """Find all positions where a room template could potentially fit"""
        valid = []

        for y in range(self.height - template.height + 1):
            for x in range(self.width - template.width + 1):
                # Check if enough buildable tiles in this area
                area = self.buildable_grid[
                    y : y + template.height, x : x + template.width
                ]
                if (
                    np.sum(area) >= template.width * template.height * 0.8
                ):  # 80% coverage required
                    valid.append((x, y))

        return valid

    def _can_place_room_at(self, template: RoomTemplate, x: int, y: int) -> bool:
        """Check if a room can be placed at a specific position"""
        # Check bounds
        if (
            x < 0
            or y < 0
            or x + template.width > self.width
            or y + template.height > self.height
        ):
            return False

        # Check for overlaps with existing rooms
        new_room = PlacedRoom(template, x, y)
        for existing_room in self.placed_rooms:
            if new_room.overlaps_with(existing_room):
                return False

        # Check if area is mostly buildable
        area = self.buildable_grid[y : y + template.height, x : x + template.width]
        if np.sum(area) < template.width * template.height * 0.8:
            return False

        # Check if area is mostly empty
        grid_area = self.grid[y : y + template.height, x : x + template.width]
        if np.sum(grid_area > 0) > template.width * template.height * 0.2:
            return False

        return True

    def _place_room_at(self, template: RoomTemplate, x: int, y: int, rotated: bool):
        """Place a room at the specified position"""
        placed_room = PlacedRoom(template, x, y, rotated)
        self.placed_rooms.append(placed_room)

        # Mark room on grid
        for row_idx, row in enumerate(template.layout):
            for col_idx, char in enumerate(row):
                grid_x = x + col_idx
                grid_y = y + row_idx

                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    if char == "#":
                        self.grid[grid_y, grid_x] = 1  # Wall
                    elif char == "D":
                        self.grid[grid_y, grid_x] = 2  # Door
                    elif char != " ":
                        self.grid[grid_y, grid_x] = 3  # Room interior

        logger.info(f"Placed {template.room_type.value} at ({x}, {y})")

    def _distance_to_room(self, position: Tuple[int, int], room: PlacedRoom) -> float:
        """Calculate distance from a position to a room"""
        x, y = position
        room_center_x = room.x + room.template.width // 2
        room_center_y = room.y + room.template.height // 2

        return ((x - room_center_x) ** 2 + (y - room_center_y) ** 2) ** 0.5

    def _add_corridors(self):
        """Add corridors to connect rooms"""
        # Simple corridor generation - connect room doors
        for i, room1 in enumerate(self.placed_rooms):
            # Find nearest unconnected room
            for room2 in self.placed_rooms[i + 1 :]:
                # Get door positions
                for door1_x, door1_y in room1.template.doors:
                    door1_world_x = room1.x + door1_x
                    door1_world_y = room1.y + door1_y

                    for door2_x, door2_y in room2.template.doors:
                        door2_world_x = room2.x + door2_x
                        door2_world_y = room2.y + door2_y

                        # Connect with L-shaped corridor
                        self._connect_points(
                            (door1_world_x, door1_world_y),
                            (door2_world_x, door2_world_y),
                        )
                        break  # Connect only one pair of doors
                    break

    def _connect_points(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Connect two points with a corridor"""
        x1, y1 = p1
        x2, y2 = p2

        # Simple L-shaped corridor
        # First go horizontal
        if x1 < x2:
            for x in range(x1, x2 + 1):
                if 0 <= x < self.width and 0 <= y1 < self.height:
                    if self.grid[y1, x] == 0 and self.buildable_grid[y1, x]:
                        self.grid[y1, x] = 4  # Corridor
        else:
            for x in range(x2, x1 + 1):
                if 0 <= x < self.width and 0 <= y1 < self.height:
                    if self.grid[y1, x] == 0 and self.buildable_grid[y1, x]:
                        self.grid[y1, x] = 4  # Corridor

        # Then go vertical
        if y1 < y2:
            for y in range(y1, y2 + 1):
                if 0 <= x2 < self.width and 0 <= y < self.height:
                    if self.grid[y, x2] == 0 and self.buildable_grid[y, x2]:
                        self.grid[y, x2] = 4  # Corridor
        else:
            for y in range(y2, y1 + 1):
                if 0 <= x2 < self.width and 0 <= y < self.height:
                    if self.grid[y, x2] == 0 and self.buildable_grid[y, x2]:
                        self.grid[y, x2] = 4  # Corridor

    def _fill_remaining_space(self):
        """Fill remaining buildable space with storage or outdoor areas"""
        # Simple fill - mark remaining buildable areas as outdoor/storage zones
        for y in range(self.height):
            for x in range(self.width):
                if self.buildable_grid[y, x] and self.grid[y, x] == 0:
                    # Check if surrounded by rooms (potential courtyard)
                    neighbors = self._count_neighbors(x, y, [1, 2, 3, 4])
                    if neighbors >= 3:
                        self.grid[y, x] = 5  # Courtyard/outdoor area

    def _count_neighbors(self, x: int, y: int, types: List[int]) -> int:
        """Count neighboring cells of specific types"""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny, nx] in types:
                        count += 1
        return count

    def to_building_grid(self) -> List[List[Optional[BuildingType]]]:
        """Convert planned base to BuildingType grid for visualization"""
        building_grid = []

        type_mapping = {
            1: BuildingType.WALL,
            2: BuildingType.DOOR,
            3: BuildingType.FLOOR,  # Room interior
            4: BuildingType.FLOOR,  # Corridor
            5: BuildingType.FLOOR,  # Courtyard
        }

        for row in self.grid:
            building_row = []
            for cell in row:
                building_row.append(type_mapping.get(cell))
            building_grid.append(building_row)

        return building_grid

    def get_summary(self) -> str:
        """Get a summary of the planned base"""
        room_counts = {}
        for room in self.placed_rooms:
            room_type = room.template.room_type.value
            room_counts[room_type] = room_counts.get(room_type, 0) + 1

        summary = "Base Plan Summary:\n"
        summary += f"  Grid size: {self.width}x{self.height}\n"
        summary += f"  Buildable tiles: {np.sum(self.buildable_grid)}\n"
        summary += f"  Rooms placed: {len(self.placed_rooms)}\n"

        for room_type, count in sorted(room_counts.items()):
            summary += f"    {room_type}: {count}\n"

        return summary

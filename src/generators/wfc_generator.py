"""
Wave Function Collapse generator for RimWorld base layouts.
Generates optimized base designs respecting building constraints and adjacency rules.
"""

import random
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from src.models.game_entities import BuildingType

logger = logging.getLogger(__name__)


class TileType(Enum):
    """Possible tile types in the WFC grid"""

    EMPTY = "empty"
    WALL = "wall"
    DOOR = "door"
    BEDROOM = "bedroom"
    STORAGE = "storage"
    WORKSHOP = "workshop"
    KITCHEN = "kitchen"
    DINING = "dining"
    RECREATION = "recreation"
    HOSPITAL = "hospital"
    RESEARCH = "research"
    POWER = "power"
    BATTERY = "battery"
    CORRIDOR = "corridor"
    OUTDOOR = "outdoor"
    FARM = "farm"


@dataclass
class Tile:
    """Represents a single tile in the WFC grid"""

    position: Tuple[int, int]
    collapsed: bool = False
    possible_types: Set[TileType] = None
    final_type: Optional[TileType] = None

    def __post_init__(self):
        if self.possible_types is None:
            self.possible_types = set(TileType)

    @property
    def entropy(self) -> int:
        """Number of possible states for this tile"""
        return len(self.possible_types)


class WFCGenerator:
    """Wave Function Collapse generator for RimWorld bases"""

    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.grid: List[List[Tile]] = []
        self.random = random.Random(seed)

        # Initialize adjacency rules
        self.adjacency_rules = self._create_adjacency_rules()

        # Initialize grid with all possibilities
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize the grid with tiles having all possible states"""
        self.grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                tile = Tile(position=(x, y))
                row.append(tile)
            self.grid.append(row)

    def _create_adjacency_rules(self) -> Dict[TileType, Dict[str, Set[TileType]]]:
        """
        Define which tile types can be adjacent to each other.
        Returns dict mapping each TileType to allowed neighbors in each direction.
        """
        rules = {}

        # Wall rules - walls connect to walls, doors, and can be next to rooms
        rules[TileType.WALL] = {
            "north": {TileType.WALL, TileType.DOOR, TileType.EMPTY, TileType.OUTDOOR},
            "south": {TileType.WALL, TileType.DOOR, TileType.EMPTY, TileType.OUTDOOR},
            "east": {TileType.WALL, TileType.DOOR, TileType.EMPTY, TileType.OUTDOOR},
            "west": {TileType.WALL, TileType.DOOR, TileType.EMPTY, TileType.OUTDOOR},
        }

        # Door rules - doors must connect to walls on sides, rooms/corridors on front/back
        rules[TileType.DOOR] = {
            "north": {
                TileType.CORRIDOR,
                TileType.BEDROOM,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.HOSPITAL,
                TileType.RESEARCH,
            },
            "south": {
                TileType.CORRIDOR,
                TileType.BEDROOM,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.HOSPITAL,
                TileType.RESEARCH,
            },
            "east": {TileType.WALL},
            "west": {TileType.WALL},
        }

        # Bedroom rules - bedrooms need walls or doors on edges
        rules[TileType.BEDROOM] = {
            "north": {TileType.WALL, TileType.DOOR, TileType.BEDROOM},
            "south": {TileType.WALL, TileType.DOOR, TileType.BEDROOM},
            "east": {TileType.WALL, TileType.DOOR, TileType.BEDROOM},
            "west": {TileType.WALL, TileType.DOOR, TileType.BEDROOM},
        }

        # Storage rules - can be adjacent to workshops, kitchens, corridors
        rules[TileType.STORAGE] = {
            "north": {
                TileType.WALL,
                TileType.DOOR,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.CORRIDOR,
            },
            "south": {
                TileType.WALL,
                TileType.DOOR,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.CORRIDOR,
            },
            "east": {
                TileType.WALL,
                TileType.DOOR,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.CORRIDOR,
            },
            "west": {
                TileType.WALL,
                TileType.DOOR,
                TileType.STORAGE,
                TileType.WORKSHOP,
                TileType.CORRIDOR,
            },
        }

        # Workshop rules - needs storage nearby, power access
        rules[TileType.WORKSHOP] = {
            "north": {
                TileType.WALL,
                TileType.DOOR,
                TileType.WORKSHOP,
                TileType.STORAGE,
                TileType.CORRIDOR,
            },
            "south": {
                TileType.WALL,
                TileType.DOOR,
                TileType.WORKSHOP,
                TileType.STORAGE,
                TileType.CORRIDOR,
            },
            "east": {
                TileType.WALL,
                TileType.DOOR,
                TileType.WORKSHOP,
                TileType.STORAGE,
                TileType.CORRIDOR,
            },
            "west": {
                TileType.WALL,
                TileType.DOOR,
                TileType.WORKSHOP,
                TileType.STORAGE,
                TileType.CORRIDOR,
            },
        }

        # Kitchen rules - should be near dining and storage
        rules[TileType.KITCHEN] = {
            "north": {
                TileType.WALL,
                TileType.DOOR,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.STORAGE,
            },
            "south": {
                TileType.WALL,
                TileType.DOOR,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.STORAGE,
            },
            "east": {
                TileType.WALL,
                TileType.DOOR,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.STORAGE,
            },
            "west": {
                TileType.WALL,
                TileType.DOOR,
                TileType.KITCHEN,
                TileType.DINING,
                TileType.STORAGE,
            },
        }

        # Dining rules - should be near kitchen and recreation
        rules[TileType.DINING] = {
            "north": {
                TileType.WALL,
                TileType.DOOR,
                TileType.DINING,
                TileType.KITCHEN,
                TileType.RECREATION,
            },
            "south": {
                TileType.WALL,
                TileType.DOOR,
                TileType.DINING,
                TileType.KITCHEN,
                TileType.RECREATION,
            },
            "east": {
                TileType.WALL,
                TileType.DOOR,
                TileType.DINING,
                TileType.KITCHEN,
                TileType.RECREATION,
            },
            "west": {
                TileType.WALL,
                TileType.DOOR,
                TileType.DINING,
                TileType.KITCHEN,
                TileType.RECREATION,
            },
        }

        # Corridor rules - connects to everything except outdoor
        rules[TileType.CORRIDOR] = {
            "north": {t for t in TileType if t != TileType.OUTDOOR},
            "south": {t for t in TileType if t != TileType.OUTDOOR},
            "east": {t for t in TileType if t != TileType.OUTDOOR},
            "west": {t for t in TileType if t != TileType.OUTDOOR},
        }

        # Power rules - can be anywhere but prefer near workshops
        rules[TileType.POWER] = {
            "north": {
                TileType.WALL,
                TileType.POWER,
                TileType.BATTERY,
                TileType.WORKSHOP,
                TileType.EMPTY,
            },
            "south": {
                TileType.WALL,
                TileType.POWER,
                TileType.BATTERY,
                TileType.WORKSHOP,
                TileType.EMPTY,
            },
            "east": {
                TileType.WALL,
                TileType.POWER,
                TileType.BATTERY,
                TileType.WORKSHOP,
                TileType.EMPTY,
            },
            "west": {
                TileType.WALL,
                TileType.POWER,
                TileType.BATTERY,
                TileType.WORKSHOP,
                TileType.EMPTY,
            },
        }

        # Fill in remaining rules with basic constraints
        for tile_type in TileType:
            if tile_type not in rules:
                # Default: can be next to walls, doors, or same type
                rules[tile_type] = {
                    "north": {TileType.WALL, TileType.DOOR, tile_type, TileType.EMPTY},
                    "south": {TileType.WALL, TileType.DOOR, tile_type, TileType.EMPTY},
                    "east": {TileType.WALL, TileType.DOOR, tile_type, TileType.EMPTY},
                    "west": {TileType.WALL, TileType.DOOR, tile_type, TileType.EMPTY},
                }

        return rules

    def get_tile(self, x: int, y: int) -> Optional[Tile]:
        """Get tile at position, returns None if out of bounds"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def get_neighbors(self, x: int, y: int) -> Dict[str, Optional[Tile]]:
        """Get neighboring tiles"""
        return {
            "north": self.get_tile(x, y - 1),
            "south": self.get_tile(x, y + 1),
            "east": self.get_tile(x + 1, y),
            "west": self.get_tile(x - 1, y),
        }

    def collapse_tile(self, tile: Tile) -> bool:
        """
        Collapse a tile to a single state.
        Returns True if successful, False if no valid states.
        """
        if tile.collapsed:
            return True

        if not tile.possible_types:
            return False

        # Choose random state from possibilities
        tile.final_type = self.random.choice(list(tile.possible_types))
        tile.collapsed = True
        tile.possible_types = {tile.final_type}

        return True

    def propagate_constraints(self, x: int, y: int) -> bool:
        """
        Propagate constraints from a collapsed tile to its neighbors.
        Returns False if contradiction found.
        """
        tile = self.get_tile(x, y)
        if not tile or not tile.collapsed:
            return True

        neighbors = self.get_neighbors(x, y)
        tile_type = tile.final_type

        # Check each neighbor
        for direction, neighbor in neighbors.items():
            if neighbor and not neighbor.collapsed:
                # Get allowed types for this direction
                allowed = self.adjacency_rules.get(tile_type, {}).get(direction, set())

                # If no specific rules, allow anything except conflicting types
                if not allowed:
                    allowed = set(TileType)

                # Constrain neighbor's possibilities
                old_possibilities = len(neighbor.possible_types)
                neighbor.possible_types &= allowed

                # Check for contradiction
                if not neighbor.possible_types:
                    # Try to relax constraints if we hit a dead end
                    if old_possibilities > 0:
                        neighbor.possible_types = {TileType.EMPTY, TileType.CORRIDOR}
                    else:
                        return False

        return True

    def find_lowest_entropy_tile(self) -> Optional[Tile]:
        """Find uncollapsed tile with lowest entropy (fewest possibilities)"""
        min_entropy = float("inf")
        candidates = []

        for row in self.grid:
            for tile in row:
                if not tile.collapsed and tile.entropy > 0:
                    if tile.entropy < min_entropy:
                        min_entropy = tile.entropy
                        candidates = [tile]
                    elif tile.entropy == min_entropy:
                        candidates.append(tile)

        if candidates:
            return self.random.choice(candidates)
        return None

    def generate(
        self, buildable_positions: Optional[Set[Tuple[int, int]]] = None
    ) -> bool:
        """
        Generate a base layout using Wave Function Collapse.

        Args:
            buildable_positions: Set of (x, y) positions that can have buildings (e.g., bridge locations)

        Returns:
            True if generation successful, False if contradiction occurred
        """
        # If buildable positions provided, constrain the grid
        if buildable_positions:
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in buildable_positions:
                        # Non-buildable areas can only be empty or outdoor
                        self.grid[y][x].possible_types = {
                            TileType.EMPTY,
                            TileType.OUTDOOR,
                        }

        # Main WFC loop
        iteration = 0
        max_iterations = self.width * self.height * 10

        while iteration < max_iterations:
            # Find tile with lowest entropy
            tile = self.find_lowest_entropy_tile()

            if not tile:
                # All tiles collapsed or no valid tiles
                break

            # Collapse the tile
            if not self.collapse_tile(tile):
                logger.warning(f"Failed to collapse tile at {tile.position}")
                return False

            # Propagate constraints
            x, y = tile.position
            if not self.propagate_constraints(x, y):
                logger.warning(f"Constraint contradiction at {tile.position}")
                return False

            iteration += 1

        # Check if all tiles are collapsed
        for row in self.grid:
            for tile in row:
                if not tile.collapsed:
                    logger.warning(f"Tile at {tile.position} not collapsed")
                    return False

        logger.info(f"Successfully generated base in {iteration} iterations")
        return True

    def to_building_grid(self) -> List[List[Optional[BuildingType]]]:
        """Convert WFC grid to BuildingType grid for visualization"""
        building_grid = []

        type_mapping = {
            TileType.WALL: BuildingType.WALL,
            TileType.DOOR: BuildingType.DOOR,
            TileType.BEDROOM: BuildingType.FURNITURE,
            TileType.STORAGE: BuildingType.STORAGE,
            TileType.WORKSHOP: BuildingType.PRODUCTION,
            TileType.KITCHEN: BuildingType.PRODUCTION,
            TileType.DINING: BuildingType.FURNITURE,
            TileType.HOSPITAL: BuildingType.FURNITURE,
            TileType.RESEARCH: BuildingType.PRODUCTION,
            TileType.POWER: BuildingType.POWER,
            TileType.BATTERY: BuildingType.POWER,
        }

        for row in self.grid:
            building_row = []
            for tile in row:
                if tile.collapsed and tile.final_type in type_mapping:
                    building_row.append(type_mapping[tile.final_type])
                else:
                    building_row.append(None)
            building_grid.append(building_row)

        return building_grid

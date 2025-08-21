"""
RimWorld Best Practices Module
Encodes proven base building strategies and rules for optimal colony design.
Includes support for Realistic Rooms mod.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import numpy as np


class ModConfig(Enum):
    """Mod configuration presets"""

    VANILLA = "vanilla"
    REALISTIC_ROOMS = "realistic_rooms"
    REALISTIC_ROOMS_LITE = "realistic_rooms_lite"


@dataclass
class RoomDimensions:
    """Room dimension requirements for different configurations"""

    # Minimum tiles to avoid cramped
    min_tiles: int
    # Standard comfortable size
    standard_tiles: int
    # Good size with mood bonus
    good_tiles: int
    # Maximum practical size
    max_practical_tiles: int

    # Suggested dimensions (width, height)
    min_dims: Tuple[int, int]
    standard_dims: Tuple[int, int]
    good_dims: Tuple[int, int]
    max_dims: Tuple[int, int]


class RimWorldBestPractices:
    """Central repository of RimWorld base building best practices"""

    # Room size configurations for different mod setups
    ROOM_CONFIGS = {
        ModConfig.VANILLA: {
            "bedroom": RoomDimensions(
                min_tiles=12,
                standard_tiles=20,
                good_tiles=30,
                max_practical_tiles=36,
                min_dims=(3, 4),
                standard_dims=(4, 5),
                good_dims=(5, 6),
                max_dims=(6, 6),
            ),
            "dining": RoomDimensions(
                min_tiles=29,
                standard_tiles=40,
                good_tiles=55,
                max_practical_tiles=70,
                min_dims=(5, 6),
                standard_dims=(6, 7),
                good_dims=(7, 8),
                max_dims=(8, 9),
            ),
            "workshop": RoomDimensions(
                min_tiles=29,
                standard_tiles=49,
                good_tiles=64,
                max_practical_tiles=81,
                min_dims=(5, 6),
                standard_dims=(7, 7),
                good_dims=(8, 8),
                max_dims=(9, 9),
            ),
            "hospital": RoomDimensions(
                min_tiles=20,
                standard_tiles=30,
                good_tiles=42,
                max_practical_tiles=56,
                min_dims=(4, 5),
                standard_dims=(5, 6),
                good_dims=(6, 7),
                max_dims=(7, 8),
            ),
            "recreation": RoomDimensions(
                min_tiles=29,
                standard_tiles=42,
                good_tiles=56,
                max_practical_tiles=72,
                min_dims=(5, 6),
                standard_dims=(6, 7),
                good_dims=(7, 8),
                max_dims=(8, 9),
            ),
            "kitchen": RoomDimensions(
                min_tiles=12,
                standard_tiles=20,
                good_tiles=30,
                max_practical_tiles=36,
                min_dims=(3, 4),
                standard_dims=(4, 5),
                good_dims=(5, 6),
                max_dims=(6, 6),
            ),
            "storage": RoomDimensions(
                min_tiles=29,
                standard_tiles=49,
                good_tiles=81,
                max_practical_tiles=121,
                min_dims=(5, 6),
                standard_dims=(7, 7),
                good_dims=(9, 9),
                max_dims=(11, 11),
            ),
            "research": RoomDimensions(
                min_tiles=20,
                standard_tiles=30,
                good_tiles=42,
                max_practical_tiles=49,
                min_dims=(4, 5),
                standard_dims=(5, 6),
                good_dims=(6, 7),
                max_dims=(7, 7),
            ),
            "prison_cell": RoomDimensions(
                min_tiles=12,
                standard_tiles=16,
                good_tiles=20,
                max_practical_tiles=25,
                min_dims=(3, 4),
                standard_dims=(4, 4),
                good_dims=(4, 5),
                max_dims=(5, 5),
            ),
            "throne": RoomDimensions(
                min_tiles=55,
                standard_tiles=81,
                good_tiles=121,
                max_practical_tiles=169,
                min_dims=(7, 8),
                standard_dims=(9, 9),
                good_dims=(11, 11),
                max_dims=(13, 13),
            ),
        },
        ModConfig.REALISTIC_ROOMS: {
            "bedroom": RoomDimensions(
                min_tiles=6,
                standard_tiles=12,
                good_tiles=16,
                max_practical_tiles=25,
                min_dims=(2, 3),
                standard_dims=(3, 4),
                good_dims=(4, 4),
                max_dims=(5, 5),
            ),
            "dining": RoomDimensions(
                min_tiles=12,
                standard_tiles=20,
                good_tiles=28,
                max_practical_tiles=45,
                min_dims=(3, 4),
                standard_dims=(4, 5),
                good_dims=(4, 7),
                max_dims=(5, 9),
            ),
            "workshop": RoomDimensions(
                min_tiles=12,
                standard_tiles=28,
                good_tiles=45,
                max_practical_tiles=75,
                min_dims=(3, 4),
                standard_dims=(4, 7),
                good_dims=(5, 9),
                max_dims=(7, 11),
            ),
            "hospital": RoomDimensions(
                min_tiles=12,
                standard_tiles=20,
                good_tiles=28,
                max_practical_tiles=45,
                min_dims=(3, 4),
                standard_dims=(4, 5),
                good_dims=(4, 7),
                max_dims=(5, 9),
            ),
            "recreation": RoomDimensions(
                min_tiles=12,
                standard_tiles=20,
                good_tiles=28,
                max_practical_tiles=45,
                min_dims=(3, 4),
                standard_dims=(4, 5),
                good_dims=(4, 7),
                max_dims=(5, 9),
            ),
            "kitchen": RoomDimensions(
                min_tiles=6,
                standard_tiles=12,
                good_tiles=16,
                max_practical_tiles=20,
                min_dims=(2, 3),
                standard_dims=(3, 4),
                good_dims=(4, 4),
                max_dims=(4, 5),
            ),
            "storage": RoomDimensions(
                min_tiles=12,
                standard_tiles=28,
                good_tiles=45,
                max_practical_tiles=75,
                min_dims=(3, 4),
                standard_dims=(4, 7),
                good_dims=(5, 9),
                max_dims=(7, 11),
            ),
            "research": RoomDimensions(
                min_tiles=12,
                standard_tiles=16,
                good_tiles=20,
                max_practical_tiles=28,
                min_dims=(3, 4),
                standard_dims=(4, 4),
                good_dims=(4, 5),
                max_dims=(4, 7),
            ),
            "prison_cell": RoomDimensions(
                min_tiles=6,
                standard_tiles=9,
                good_tiles=12,
                max_practical_tiles=16,
                min_dims=(2, 3),
                standard_dims=(3, 3),
                good_dims=(3, 4),
                max_dims=(4, 4),
            ),
            "throne": RoomDimensions(
                min_tiles=28,
                standard_tiles=45,
                good_tiles=75,
                max_practical_tiles=100,
                min_dims=(4, 7),
                standard_dims=(5, 9),
                good_dims=(7, 11),
                max_dims=(10, 10),
            ),
        },
        ModConfig.REALISTIC_ROOMS_LITE: {
            "bedroom": RoomDimensions(
                min_tiles=9,
                standard_tiles=16,
                good_tiles=20,
                max_practical_tiles=30,
                min_dims=(3, 3),
                standard_dims=(4, 4),
                good_dims=(4, 5),
                max_dims=(5, 6),
            ),
            "dining": RoomDimensions(
                min_tiles=20,
                standard_tiles=30,
                good_tiles=45,
                max_practical_tiles=60,
                min_dims=(4, 5),
                standard_dims=(5, 6),
                good_dims=(5, 9),
                max_dims=(6, 10),
            ),
            "workshop": RoomDimensions(
                min_tiles=20,
                standard_tiles=36,
                good_tiles=49,
                max_practical_tiles=64,
                min_dims=(4, 5),
                standard_dims=(6, 6),
                good_dims=(7, 7),
                max_dims=(8, 8),
            ),
            "hospital": RoomDimensions(
                min_tiles=16,
                standard_tiles=25,
                good_tiles=36,
                max_practical_tiles=49,
                min_dims=(4, 4),
                standard_dims=(5, 5),
                good_dims=(6, 6),
                max_dims=(7, 7),
            ),
            "recreation": RoomDimensions(
                min_tiles=20,
                standard_tiles=30,
                good_tiles=45,
                max_practical_tiles=60,
                min_dims=(4, 5),
                standard_dims=(5, 6),
                good_dims=(5, 9),
                max_dims=(6, 10),
            ),
            "kitchen": RoomDimensions(
                min_tiles=9,
                standard_tiles=16,
                good_tiles=20,
                max_practical_tiles=25,
                min_dims=(3, 3),
                standard_dims=(4, 4),
                good_dims=(4, 5),
                max_dims=(5, 5),
            ),
            "storage": RoomDimensions(
                min_tiles=20,
                standard_tiles=36,
                good_tiles=64,
                max_practical_tiles=100,
                min_dims=(4, 5),
                standard_dims=(6, 6),
                good_dims=(8, 8),
                max_dims=(10, 10),
            ),
            "research": RoomDimensions(
                min_tiles=16,
                standard_tiles=20,
                good_tiles=25,
                max_practical_tiles=36,
                min_dims=(4, 4),
                standard_dims=(4, 5),
                good_dims=(5, 5),
                max_dims=(6, 6),
            ),
            "prison_cell": RoomDimensions(
                min_tiles=9,
                standard_tiles=12,
                good_tiles=16,
                max_practical_tiles=20,
                min_dims=(3, 3),
                standard_dims=(3, 4),
                good_dims=(4, 4),
                max_dims=(4, 5),
            ),
            "throne": RoomDimensions(
                min_tiles=45,
                standard_tiles=64,
                good_tiles=100,
                max_practical_tiles=144,
                min_dims=(5, 9),
                standard_dims=(8, 8),
                good_dims=(10, 10),
                max_dims=(12, 12),
            ),
        },
    }

    # Adjacency rules (what rooms should be near each other)
    ADJACENCY_RULES = {
        "kitchen": ["freezer", "dining"],
        "dining": ["kitchen", "recreation"],
        "freezer": ["kitchen"],
        "hospital": ["entrance", "storage_medicine"],
        "bedroom": ["corridor"],  # Not central
        "workshop": ["storage", "workshop"],  # Cluster workshops
        "prison_cell": ["prison_cell"],  # Group prison cells
        "recreation": ["dining", "corridor"],  # Central location
        "research": ["storage"],  # Near materials
        "throne": ["recreation", "dining"],  # Central/impressive location
    }

    # Room placement priorities (1 = highest)
    PLACEMENT_PRIORITY = {
        "hospital": 1,  # Near entrance for rescue
        "kitchen": 2,  # Central for efficiency
        "dining": 2,
        "freezer": 2,
        "recreation": 3,  # Central for mood
        "workshop": 4,  # Can be more peripheral
        "storage": 4,
        "research": 5,
        "bedroom": 6,  # Edge placement
        "prison_cell": 7,  # Far from colonists
        "throne": 3,  # Central but not critical
    }

    # Traffic flow rules
    TRAFFIC_RULES = {
        "minimize_bedroom_travel": "Place bedrooms on colony edges",
        "central_hub": "Kitchen/dining/rec room should form central hub",
        "workshop_cluster": "All workshops in one area with shared toolboxes",
        "hospital_access": "Hospital near main entrance",
        "prison_isolation": "Prison section separated from colonist areas",
    }

    # Killbox specifications
    KILLBOX_SPECS = {
        "entry_width": (1, 2),  # 1-2 tiles wide entrance
        "killbox_length": (12, 20),  # Length of kill corridor
        "killbox_width": (3, 5),  # Width for defender positioning
        "turret_spacing": 3,  # Minimum tiles between turrets
        "material": "stone",  # Fire-resistant
        "avoid_paving": True,  # Don't pave (slows enemies)
        "sandbag_positions": True,
        "overhead_mountain": False,  # Avoid drop pods
    }

    # Workshop optimization
    WORKSHOP_RULES = {
        "benches_per_toolbox": 6,  # Up to 6 benches around 2 toolboxes
        "toolbox_pattern": "circular",  # Arrange in circle
        "include_benches": [
            "TableStonecutter",
            "ElectricSmithy",
            "TableMachining",
            "FabricationBench",
            "TableSculpting",
            "DrugLab",
        ],
        "shared_storage": True,  # Central material storage
    }

    # Material priorities
    MATERIAL_PRIORITY = {
        "walls_exterior": ["granite", "limestone", "sandstone"],
        "walls_interior": ["wood", "steel"],  # Faster to build
        "floors_bedroom": ["carpet", "wood"],  # Beauty and comfort
        "floors_workshop": ["concrete", "stone"],  # Durability
        "floors_hospital": ["sterile_tile"],  # Cleanliness
        "floors_kitchen": ["sterile_tile", "tile"],  # Cleanliness
        "floors_storage": ["concrete"],  # Just needs to be clean
    }

    @staticmethod
    def get_room_dimensions(
        room_type: str,
        mod_config: ModConfig = ModConfig.REALISTIC_ROOMS,
        size_preference: str = "standard",
    ) -> Tuple[int, int]:
        """
        Get recommended room dimensions.

        Args:
            room_type: Type of room (bedroom, kitchen, etc.)
            mod_config: Which mod configuration to use
            size_preference: "min", "standard", "good", or "max"

        Returns:
            Tuple of (width, height) in tiles
        """
        if room_type not in RimWorldBestPractices.ROOM_CONFIGS[mod_config]:
            # Default to bedroom dimensions if room type unknown
            room_type = "bedroom"

        room_config = RimWorldBestPractices.ROOM_CONFIGS[mod_config][room_type]

        if size_preference == "min":
            return room_config.min_dims
        elif size_preference == "good":
            return room_config.good_dims
        elif size_preference == "max":
            return room_config.max_dims
        else:  # standard
            return room_config.standard_dims

    @staticmethod
    def get_adjacency_score(room1: str, room2: str) -> float:
        """
        Get adjacency score for two room types.
        Higher score means they should be closer together.

        Args:
            room1: First room type
            room2: Second room type

        Returns:
            Score from 0 (no preference) to 1 (must be adjacent)
        """
        rules = RimWorldBestPractices.ADJACENCY_RULES

        # Check direct adjacency rules
        if room1 in rules and room2 in rules[room1]:
            return 1.0
        if room2 in rules and room1 in rules[room2]:
            return 1.0

        # Special cases
        if room1 == "bedroom" and room2 == "bedroom":
            return 0.5  # Bedrooms can be near each other
        if "workshop" in room1 and "workshop" in room2:
            return 0.8  # Workshops should cluster
        if "storage" in room1 and "workshop" in room2:
            return 0.7
        if "storage" in room2 and "workshop" in room1:
            return 0.7

        # Default: no preference
        return 0.0

    @staticmethod
    def validate_base_layout(
        grid: np.ndarray, rooms: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Validate a base layout against best practices.

        Args:
            grid: 2D numpy array of the base layout
            rooms: List of room dictionaries with type and position

        Returns:
            Dictionary of validation results and warnings
        """
        warnings = []
        suggestions = []

        # Check bedroom placement
        bedroom_positions = [r for r in rooms if r["type"] == "bedroom"]
        if bedroom_positions:
            # Check if bedrooms are on edges
            center_y, center_x = grid.shape[0] // 2, grid.shape[1] // 2
            for bedroom in bedroom_positions:
                dist_to_center = abs(bedroom["y"] - center_y) + abs(
                    bedroom["x"] - center_x
                )
                if dist_to_center < min(grid.shape) // 4:
                    warnings.append(
                        "Bedroom placed too centrally - should be on colony edges"
                    )
                    break

        # Check kitchen-freezer adjacency
        kitchen = next((r for r in rooms if r["type"] == "kitchen"), None)
        freezer = next((r for r in rooms if r["type"] == "freezer"), None)
        if kitchen and freezer:
            dist = abs(kitchen["x"] - freezer["x"]) + abs(kitchen["y"] - freezer["y"])
            if dist > 2:
                warnings.append("Kitchen and freezer should be adjacent")

        # Check hospital placement
        hospital = next((r for r in rooms if r["type"] == "hospital"), None)
        if hospital:
            # Check if near edge (for entrance access)
            edge_dist = min(
                hospital["x"],
                hospital["y"],
                grid.shape[1] - hospital["x"],
                grid.shape[0] - hospital["y"],
            )
            if edge_dist > 10:
                suggestions.append("Consider placing hospital closer to entrance")

        # Check workshop clustering
        workshops = [r for r in rooms if "workshop" in r["type"]]
        if len(workshops) > 1:
            # Calculate average distance between workshops
            total_dist = 0
            count = 0
            for i, w1 in enumerate(workshops):
                for w2 in workshops[i + 1 :]:
                    total_dist += abs(w1["x"] - w2["x"]) + abs(w1["y"] - w2["y"])
                    count += 1
            if count > 0 and total_dist / count > 10:
                suggestions.append(
                    "Workshops should be clustered together for efficiency"
                )

        return {
            "warnings": warnings,
            "suggestions": suggestions,
            "valid": len(warnings) == 0,
        }

    @staticmethod
    def optimize_traffic_flow(room_layout: Dict) -> Dict:
        """
        Optimize room placement for traffic flow.

        Args:
            room_layout: Dictionary of room positions and types

        Returns:
            Optimized room layout
        """
        # This would implement pathfinding optimization
        # For now, return the original layout
        return room_layout

    @staticmethod
    def generate_killbox_layout(width: int = 15, height: int = 20) -> np.ndarray:
        """
        Generate a standard killbox layout.

        Args:
            width: Width of killbox area
            height: Height of killbox area

        Returns:
            2D array with killbox design
        """
        from src.generators.wfc_generator import TileType

        killbox = np.zeros((height, width), dtype=int)

        specs = RimWorldBestPractices.KILLBOX_SPECS
        entry_width = specs["entry_width"][0]  # Use minimum for maximum effectiveness

        # Create funnel entrance
        entrance_x = width // 2
        for y in range(3):
            for x in range(width):
                if abs(x - entrance_x) > entry_width // 2:
                    killbox[y, x] = TileType.WALL

        # Create kill corridor
        for y in range(3, height - 5):
            for x in range(width):
                if x == 0 or x == width - 1:
                    killbox[y, x] = TileType.WALL
                elif x < 3 or x > width - 4:
                    # Defender positions
                    if y % 2 == 0:
                        killbox[y, x] = TileType.FURNITURE  # Sandbags

        # Create wider defensive area at back
        for y in range(height - 5, height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == height - 1:
                    killbox[y, x] = TileType.WALL
                elif (
                    x % specs["turret_spacing"] == 0
                    and y == height - 3
                    and 2 < x < width - 3
                ):
                    killbox[y, x] = TileType.POWER  # Turret positions

        return killbox

    @staticmethod
    def calculate_base_efficiency_score(layout: Dict) -> float:
        """
        Calculate efficiency score for a base layout.

        Args:
            layout: Dictionary describing base layout

        Returns:
            Efficiency score from 0 to 100
        """
        score = 100.0

        # Deduct points for various inefficiencies
        # This is a simplified scoring system

        # Check bedroom-workplace-dining triangle
        if "bedroom_distance" in layout:
            if layout["bedroom_distance"] > 20:
                score -= 10
            elif layout["bedroom_distance"] > 30:
                score -= 20

        # Check kitchen-freezer adjacency
        if "kitchen_freezer_distance" in layout:
            if layout["kitchen_freezer_distance"] > 1:
                score -= 15

        # Check workshop clustering
        if "workshop_spread" in layout:
            if layout["workshop_spread"] > 15:
                score -= 10

        # Check hospital accessibility
        if "hospital_edge_distance" in layout:
            if layout["hospital_edge_distance"] > 10:
                score -= 5

        return max(0, score)

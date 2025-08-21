"""
Room templates for RimWorld base generation.
Defines common room patterns and layouts.
Updated for Realistic Rooms mod dimensions.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RoomType(Enum):
    """Types of rooms in RimWorld - optimized for Realistic Rooms mod"""

    # Bedrooms (Realistic Rooms: 6 tiles min, 12 standard, 16 good)
    BEDROOM_SMALL = "bedroom_small"  # 2x3 internal (6 tiles) - minimum viable
    BEDROOM_MEDIUM = "bedroom_medium"  # 3x4 internal (12 tiles) - standard
    BEDROOM_LARGE = "bedroom_large"  # 4x4 internal (16 tiles) - good mood
    BEDROOM_LUXURY = "bedroom_luxury"  # 5x5 internal (25 tiles) - spacious
    BARRACKS = "barracks"  # 5x6 internal (30 tiles), multiple beds

    # Kitchen & Dining (Realistic Rooms: smaller is fine)
    KITCHEN = "kitchen"  # 3x4 internal (12 tiles) - standard size
    DINING_HALL = "dining_hall"  # 4x5 internal (20 tiles) - good size
    FREEZER = "freezer"  # 3x3 internal (9 tiles) - compact

    # Workshops (Realistic Rooms: 28 tiles standard)
    WORKSHOP_SMALL = "workshop_small"  # 4x4 internal (16 tiles)
    WORKSHOP = "workshop"  # 4x7 internal (28 tiles) - standard
    WORKSHOP_LARGE = "workshop_large"  # 5x9 internal (45 tiles) - multiple benches

    # Medical (Near entrance)
    HOSPITAL = "hospital"  # 4x5 internal (20 tiles) - standard

    # Research
    RESEARCH_LAB = "research_lab"  # 4x4 internal (16 tiles) - standard

    # Storage
    STORAGE_SMALL = "storage_small"  # 3x3 internal (9 tiles)
    STORAGE_MEDIUM = "storage_medium"  # 4x5 internal (20 tiles)
    STORAGE_LARGE = "storage_large"  # 5x7 internal (35 tiles)

    # Power
    POWER_ROOM = "power_room"  # 3x3 internal (9 tiles), batteries
    GENERATOR_ROOM = "generator_room"  # 3x4 internal (12 tiles)

    # Recreation (Central location)
    REC_ROOM = "rec_room"  # 4x5 internal (20 tiles) - standard

    # Prison (Isolated)
    PRISON_CELL = "prison_cell"  # 3x3 internal (9 tiles) - standard

    # Killbox components
    KILLBOX_ENTRANCE = "killbox_entrance"  # 1-2 wide funnel
    KILLBOX_CORRIDOR = "killbox_corridor"  # Kill zone

    # Traffic
    CORRIDOR = "corridor"  # 2 wide
    HALLWAY = "hallway"  # 3 wide


@dataclass
class RoomTemplate:
    """Template for a room layout"""

    room_type: RoomType
    width: int  # External dimensions including walls
    height: int
    layout: List[str]  # String representation of the room
    doors: List[Tuple[int, int]]  # Door positions relative to room
    required_items: List[str]  # Required furniture/items
    priority: int = 5  # 1-10, higher = more important

    def rotate_90(self) -> "RoomTemplate":
        """Rotate the room template 90 degrees clockwise"""
        # Transpose and reverse for 90-degree rotation
        rotated_layout = [
            "".join(row[i] for row in reversed(self.layout))
            for i in range(len(self.layout[0]))
        ]

        # Rotate door positions
        rotated_doors = [(self.height - 1 - y, x) for x, y in self.doors]

        return RoomTemplate(
            room_type=self.room_type,
            width=self.height,
            height=self.width,
            layout=rotated_layout,
            doors=rotated_doors,
            required_items=self.required_items,
            priority=self.priority,
        )


class RoomTemplateLibrary:
    """Library of room templates for base generation"""

    def __init__(self):
        self.templates = self._create_templates()

    def _create_templates(self) -> Dict[RoomType, RoomTemplate]:
        """Create all room templates"""
        templates = {}

        # Tiny bedroom (2x2 internal, 4x4 with walls)
        templates[RoomType.BEDROOM_SMALL] = RoomTemplate(
            room_type=RoomType.BEDROOM_SMALL,
            width=4,
            height=4,
            layout=["####", "#.B#", "#..#", "#D##"],
            doors=[(1, 3)],
            required_items=["Bed"],
            priority=8,
        )

        # Medium bedroom (4x4 internal, 6x6 with walls)
        templates[RoomType.BEDROOM_MEDIUM] = RoomTemplate(
            room_type=RoomType.BEDROOM_MEDIUM,
            width=6,
            height=6,
            layout=["######", "#....#", "#.BB.#", "#.TT.#", "#....#", "###D##"],
            doors=[(3, 5)],
            required_items=["DoubleBed", "EndTable", "Dresser"],
            priority=7,
        )

        # Compact kitchen (3x3 internal, 5x5 with walls)
        templates[RoomType.KITCHEN] = RoomTemplate(
            room_type=RoomType.KITCHEN,
            width=5,
            height=5,
            layout=["#####", "#SS.#", "#...#", "#C..#", "##D##"],
            doors=[(2, 4)],
            required_items=["ElectricStove", "Counter"],
            priority=9,
        )

        # Dining hall (6x6 internal, 8x8 with walls)
        templates[RoomType.DINING_HALL] = RoomTemplate(
            room_type=RoomType.DINING_HALL,
            width=8,
            height=8,
            layout=[
                "########",
                "#......#",
                "#.TTTT.#",
                "#.CCCC.#",
                "#.CCCC.#",
                "#.TTTT.#",
                "#......#",
                "###DD###",
            ],
            doors=[(3, 7), (4, 7)],
            required_items=["Table", "DiningChair"],
            priority=8,
        )

        # Workshop (5x5 internal, 7x7 with walls)
        templates[RoomType.WORKSHOP] = RoomTemplate(
            room_type=RoomType.WORKSHOP,
            width=7,
            height=7,
            layout=[
                "#######",
                "#.....#",
                "#.WWW.#",
                "#.....#",
                "#.BBB.#",
                "#.....#",
                "###D###",
            ],
            doors=[(3, 6)],
            required_items=[
                "ElectricSmithy",
                "ElectricTailoringBench",
                "TableMachining",
            ],
            priority=7,
        )

        # Storage room (6x6 internal, 8x8 with walls)
        templates[RoomType.STORAGE_LARGE] = RoomTemplate(
            room_type=RoomType.STORAGE_LARGE,
            width=8,
            height=8,
            layout=[
                "########",
                "#SSSSSS#",
                "#......#",
                "#......#",
                "#......#",
                "#......#",
                "#SSSSSS#",
                "###DD###",
            ],
            doors=[(3, 7), (4, 7)],
            required_items=["Shelf"],
            priority=6,
        )

        # Hospital (5x6 internal, 7x8 with walls)
        templates[RoomType.HOSPITAL] = RoomTemplate(
            room_type=RoomType.HOSPITAL,
            width=7,
            height=8,
            layout=[
                "#######",
                "#.....#",
                "#.HHH.#",
                "#.....#",
                "#.HHH.#",
                "#.....#",
                "#.VVV.#",
                "###D###",
            ],
            doors=[(3, 7)],
            required_items=["HospitalBed", "VitalMonitor"],
            priority=7,
        )

        # Power/Battery room (3x4 internal, 5x6 with walls)
        templates[RoomType.POWER_ROOM] = RoomTemplate(
            room_type=RoomType.POWER_ROOM,
            width=5,
            height=6,
            layout=["#####", "#BBB#", "#...#", "#...#", "#BBB#", "##D##"],
            doors=[(2, 5)],
            required_items=["Battery"],
            priority=9,
        )

        # Recreation room (5x5 internal, 7x7 with walls)
        templates[RoomType.REC_ROOM] = RoomTemplate(
            room_type=RoomType.REC_ROOM,
            width=7,
            height=7,
            layout=[
                "#######",
                "#.....#",
                "#.PPP.#",
                "#.....#",
                "#.CCC.#",
                "#.....#",
                "###D###",
            ],
            doors=[(3, 6)],
            required_items=["BilliardsTable", "ChessTable", "Television"],
            priority=5,
        )

        # Freezer (4x4 internal, 6x6 with walls)
        templates[RoomType.FREEZER] = RoomTemplate(
            room_type=RoomType.FREEZER,
            width=6,
            height=6,
            layout=["######", "#CCCC#", "#....#", "#....#", "#....#", "##DD##"],
            doors=[(2, 5), (3, 5)],  # Double door for hauling
            required_items=["Cooler"],
            priority=8,
        )

        return templates

    def get_template(self, room_type: RoomType) -> Optional[RoomTemplate]:
        """Get a specific room template"""
        return self.templates.get(room_type)

    def get_templates_by_priority(self, min_priority: int = 0) -> List[RoomTemplate]:
        """Get all templates sorted by priority"""
        return sorted(
            [t for t in self.templates.values() if t.priority >= min_priority],
            key=lambda x: x.priority,
            reverse=True,
        )

    def get_templates_fitting_space(
        self, width: int, height: int
    ) -> List[RoomTemplate]:
        """Get all templates that fit within given dimensions"""
        fitting = []
        for template in self.templates.values():
            # Check both orientations
            if template.width <= width and template.height <= height:
                fitting.append(template)
            # Check rotated version
            if template.height <= width and template.width <= height:
                fitting.append(template.rotate_90())

        return sorted(fitting, key=lambda x: x.priority, reverse=True)


# Legend for room layout strings:
# # = Wall
# D = Door
# . = Empty floor
# B = Bed
# T = Table
# C = Chair/Counter/Cooler (context dependent)
# S = Storage/Shelf/Stove (context dependent)
# W = Workbench
# H = Hospital bed
# V = Vital monitor
# P = Production/Play equipment (context dependent)

"""
Realistic RimWorld base generator that produces actual room layouts with furniture.
Based on analysis of real RimWorld bases and AlphaPrefabs patterns.
"""

import numpy as np
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict, Optional


class CellType(IntEnum):
    """Cell types that match actual RimWorld objects"""
    EMPTY = 0
    WALL = 1
    DOOR = 2
    BED = 3
    TABLE = 4
    CHAIR = 5
    STORAGE = 6
    WORKBENCH = 7
    STOVE = 8
    TORCH = 9
    FLOOR = 10
    DRESSER = 11
    ENDTABLE = 12
    RECREATION = 13
    MEDICAL_BED = 14
    RESEARCH_BENCH = 15
    BATTERY = 16
    SOLAR_PANEL = 17
    WIND_TURBINE = 18
    COOLER = 19
    HEATER = 20
    PLANT_POT = 21
    SANDBAG = 22
    TURRET = 23


@dataclass
class Room:
    """Represents a room with position and contents"""
    x: int
    y: int
    width: int
    height: int
    room_type: str
    contents: np.ndarray = None
    
    def __post_init__(self):
        if self.contents is None:
            self.contents = np.zeros((self.height, self.width), dtype=int)


class RealisticBaseGenerator:
    """Generates realistic RimWorld bases with actual furniture and layouts"""
    
    # Room templates based on actual RimWorld designs
    BEDROOM_TEMPLATES = [
        # Small efficient bedroom (5x6)
        {
            "size": (5, 6),
            "layout": [
                "WWDWW",
                "WFFFW",
                "WBFFW",
                "WEFFW",
                "WRRFW",
                "WWWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "B": CellType.BED,
                "E": CellType.ENDTABLE,
                "R": CellType.DRESSER,
                "F": CellType.FLOOR,
                "T": CellType.TORCH
            }
        },
        # Medium bedroom (6x7)
        {
            "size": (6, 7),
            "layout": [
                "WWWWWW",
                "WFFFFW",
                "WBEFRW",
                "WFFFFW",
                "WFFFFW",
                "WFTFFW",
                "WWDWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "B": CellType.BED,
                "E": CellType.ENDTABLE,
                "R": CellType.DRESSER,
                "F": CellType.FLOOR,
                "T": CellType.TABLE
            }
        }
    ]
    
    KITCHEN_TEMPLATES = [
        # Efficient kitchen (7x6)
        {
            "size": (7, 6),
            "layout": [
                "WWWDWWW",
                "WSFFFFW",
                "WSFFFFW",
                "WTFCFFW",
                "WFFTFFW",
                "WWWWWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "S": CellType.STOVE,
                "T": CellType.TABLE,
                "C": CellType.CHAIR,
                "F": CellType.FLOOR
            }
        }
    ]
    
    WORKSHOP_TEMPLATES = [
        # Production room (8x7)
        {
            "size": (8, 7),
            "layout": [
                "WWWWWWWW",
                "WBBBFFFW",
                "DWFFFFFW",
                "WBBBFFFW",
                "WSSFFFFW",
                "WSSFFFFW",
                "WWWWWWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "B": CellType.WORKBENCH,
                "S": CellType.STORAGE,
                "F": CellType.FLOOR
            }
        }
    ]
    
    STORAGE_TEMPLATES = [
        # Storage room (6x6)
        {
            "size": (6, 6),
            "layout": [
                "WWWWWW",
                "WSSSFW",
                "WSSSFW",
                "WSSSFW",
                "WFFFFW",
                "WWDWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "S": CellType.STORAGE,
                "F": CellType.FLOOR
            }
        }
    ]
    
    DINING_TEMPLATES = [
        # Dining hall (8x8)
        {
            "size": (8, 8),
            "layout": [
                "WWWDWWWW",
                "WFFFFFFW",
                "WCTTTCFW",
                "WFFFFFFW",
                "WCTTTCFW",
                "WFFFFFFW",
                "WFFFFFPW",
                "WWWWWWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "T": CellType.TABLE,
                "C": CellType.CHAIR,
                "F": CellType.FLOOR,
                "P": CellType.PLANT_POT
            }
        }
    ]
    
    HOSPITAL_TEMPLATES = [
        # Medical room (7x6)
        {
            "size": (7, 6),
            "layout": [
                "WWWWWWW",
                "WMFFFMW",
                "WEFFFEW",
                "WMFFFMW",
                "WFFFFFW",
                "WWWDWWW"
            ],
            "legend": {
                "W": CellType.WALL,
                "D": CellType.DOOR,
                "M": CellType.MEDICAL_BED,
                "E": CellType.ENDTABLE,
                "F": CellType.FLOOR
            }
        }
    ]
    
    def __init__(self, width: int = 100, height: int = 100):
        """Initialize the generator with map dimensions"""
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.rooms: List[Room] = []
        
    def parse_template(self, template: dict) -> np.ndarray:
        """Convert a template dictionary to a numpy array"""
        width, height = template["size"]
        layout = template["layout"]
        legend = template["legend"]
        
        result = np.zeros((height, width), dtype=int)
        for y, row in enumerate(layout):
            for x, char in enumerate(row):
                if x < width and y < height:  # Bounds check
                    if char in legend:
                        result[y, x] = legend[char].value
                    
        return result
    
    def place_room(self, room_type: str, x: int, y: int) -> bool:
        """Place a room at the specified position"""
        # Select appropriate template
        templates_map = {
            "bedroom": self.BEDROOM_TEMPLATES,
            "kitchen": self.KITCHEN_TEMPLATES,
            "workshop": self.WORKSHOP_TEMPLATES,
            "storage": self.STORAGE_TEMPLATES,
            "dining": self.DINING_TEMPLATES,
            "hospital": self.HOSPITAL_TEMPLATES
        }
        
        if room_type not in templates_map:
            return False
            
        templates = templates_map[room_type]
        template = random.choice(templates)
        room_grid = self.parse_template(template)
        
        height, width = room_grid.shape
        
        # Check if room fits
        if x + width > self.width or y + height > self.height:
            return False
            
        # Check for overlaps (only with non-empty cells)
        for dy in range(height):
            for dx in range(width):
                if room_grid[dy, dx] != CellType.EMPTY:
                    if self.grid[y + dy, x + dx] != CellType.EMPTY:
                        return False
        
        # Place the room
        for dy in range(height):
            for dx in range(width):
                if room_grid[dy, dx] != CellType.EMPTY:
                    self.grid[y + dy, x + dx] = room_grid[dy, dx]
                    
        # Record room placement
        room = Room(x, y, width, height, room_type, room_grid)
        self.rooms.append(room)
        
        return True
    
    def add_corridors(self):
        """Add corridors to connect rooms"""
        # Create a main corridor system
        if not self.rooms:
            return
            
        # Find center of all rooms
        center_x = sum(r.x + r.width // 2 for r in self.rooms) // len(self.rooms)
        center_y = sum(r.y + r.height // 2 for r in self.rooms) // len(self.rooms)
        
        # Connect each room to the central corridor
        for room in self.rooms:
            door_pos = self.find_door_in_room(room)
            if door_pos:
                # Connect door to center point
                self.connect_points(door_pos, (center_x, center_y))
                
        # Also connect adjacent rooms directly
        for i, room1 in enumerate(self.rooms):
            if i + 1 < len(self.rooms):
                room2 = self.rooms[i + 1]
                
                # Find door positions
                door1_pos = self.find_door_in_room(room1)
                door2_pos = self.find_door_in_room(room2)
                
                if door1_pos and door2_pos:
                    # Only connect if rooms are close
                    dist = abs(door1_pos[0] - door2_pos[0]) + abs(door1_pos[1] - door2_pos[1])
                    if dist < 20:  # Only connect nearby rooms
                        self.connect_points(door1_pos, door2_pos)
    
    def find_door_in_room(self, room: Room) -> Optional[Tuple[int, int]]:
        """Find door position in a room"""
        for y in range(room.height):
            for x in range(room.width):
                if room.contents[y, x] == CellType.DOOR:
                    return (room.x + x, room.y + y)
        return None
    
    def connect_points(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Connect two points with a corridor"""
        x1, y1 = p1
        x2, y2 = p2
        
        # Simple L-shaped corridor
        # Horizontal first
        if x1 < x2:
            for x in range(x1, x2 + 1):
                if self.grid[y1, x] == CellType.EMPTY:
                    self.grid[y1, x] = CellType.FLOOR
        else:
            for x in range(x2, x1 + 1):
                if self.grid[y1, x] == CellType.EMPTY:
                    self.grid[y1, x] = CellType.FLOOR
        
        # Then vertical
        if y1 < y2:
            for y in range(y1, y2 + 1):
                if self.grid[y, x2] == CellType.EMPTY:
                    self.grid[y, x2] = CellType.FLOOR
        else:
            for y in range(y2, y1 + 1):
                if self.grid[y, x2] == CellType.EMPTY:
                    self.grid[y, x2] = CellType.FLOOR
    
    def add_perimeter_wall(self):
        """Add a perimeter wall around the base"""
        # Find the bounding box of all rooms
        if not self.rooms:
            return
            
        min_x = min(r.x for r in self.rooms) - 2
        max_x = max(r.x + r.width for r in self.rooms) + 2
        min_y = min(r.y for r in self.rooms) - 2
        max_y = max(r.y + r.height for r in self.rooms) + 2
        
        # Clamp to grid bounds
        min_x = max(0, min_x)
        max_x = min(self.width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(self.height - 1, max_y)
        
        # Add walls
        for x in range(min_x, max_x + 1):
            if self.grid[min_y, x] == CellType.EMPTY:
                self.grid[min_y, x] = CellType.WALL
            if self.grid[max_y, x] == CellType.EMPTY:
                self.grid[max_y, x] = CellType.WALL
                
        for y in range(min_y, max_y + 1):
            if self.grid[y, min_x] == CellType.EMPTY:
                self.grid[y, min_x] = CellType.WALL
            if self.grid[y, max_x] == CellType.EMPTY:
                self.grid[y, max_x] = CellType.WALL
    
    def add_entrance(self):
        """Add main entrance to the base"""
        # Find a wall position for entrance
        wall_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == CellType.WALL:
                    # Check if it's an exterior wall
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                if self.grid[ny, nx] != CellType.EMPTY:
                                    neighbors += 1
                    if neighbors < 5:  # Exterior wall
                        wall_positions.append((x, y))
        
        if wall_positions:
            entrance = random.choice(wall_positions)
            self.grid[entrance[1], entrance[0]] = CellType.DOOR
    
    def generate_base(self, num_bedrooms: int = 5, 
                     include_kitchen: bool = True,
                     include_dining: bool = True,
                     include_workshop: int = 2,
                     include_storage: bool = True,
                     include_hospital: bool = True) -> np.ndarray:
        """Generate a complete base layout"""
        
        # Reset grid
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.rooms = []
        
        # Calculate starting position (more centered)
        start_x = self.width // 2 - 15  # Start more centered
        start_y = self.height // 2 - 15
        
        # Room placement order and spacing
        current_x = start_x
        current_y = start_y
        row_height = 0
        spacing = 1  # Tighter spacing for connected base
        
        # Place bedrooms
        for i in range(num_bedrooms):
            placed = False
            attempts = 0
            while not placed and attempts < 10:
                if self.place_room("bedroom", current_x, current_y):
                    placed = True
                    room = self.rooms[-1]
                    current_x += room.width + spacing
                    row_height = max(row_height, room.height)
                    
                    # Start new row if needed
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += row_height + spacing
                        row_height = 0
                else:
                    current_x += 8
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += 10
                attempts += 1
        
        # Place kitchen
        if include_kitchen:
            placed = False
            attempts = 0
            while not placed and attempts < 10:
                if self.place_room("kitchen", current_x, current_y):
                    placed = True
                    room = self.rooms[-1]
                    current_x += room.width + spacing
                else:
                    current_x += 8
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += 10
                attempts += 1
        
        # Place dining
        if include_dining:
            placed = False
            attempts = 0
            while not placed and attempts < 10:
                if self.place_room("dining", current_x, current_y):
                    placed = True
                    room = self.rooms[-1]
                    current_x += room.width + spacing
                else:
                    current_x += 8
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += 10
                attempts += 1
        
        # Place workshops
        for i in range(include_workshop):
            placed = False
            attempts = 0
            while not placed and attempts < 10:
                if self.place_room("workshop", current_x, current_y):
                    placed = True
                    room = self.rooms[-1]
                    current_x += room.width + spacing
                    
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += row_height + spacing
                else:
                    current_x += 8
                    if current_x > self.width - 20:
                        current_x = start_x
                        current_y += 10
                attempts += 1
        
        # Place storage
        if include_storage:
            self.place_room("storage", current_x, current_y)
            
        # Place hospital
        if include_hospital:
            current_x = start_x
            current_y += 10
            self.place_room("hospital", current_x, current_y)
        
        # Add corridors between rooms
        self.add_corridors()
        
        # Add perimeter wall
        self.add_perimeter_wall()
        
        # Add entrance
        self.add_entrance()
        
        return self.grid
    
    def get_description(self) -> str:
        """Get a description of the generated base"""
        room_counts = {}
        for room in self.rooms:
            if room.room_type in room_counts:
                room_counts[room.room_type] += 1
            else:
                room_counts[room.room_type] = 1
        
        lines = ["Generated realistic RimWorld base:"]
        for room_type, count in room_counts.items():
            lines.append(f"  - {count} {room_type}(s)")
        lines.append(f"Total rooms: {len(self.rooms)}")
        
        return "\n".join(lines)
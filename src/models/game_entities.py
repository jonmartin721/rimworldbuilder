from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class ThingCategory(str, Enum):
    BUILDING = "Building"
    PAWN = "Pawn"
    PLANT = "Plant"
    ITEM = "Item"
    FILTH = "Filth"
    TERRAIN = "Terrain"
    ZONE = "Zone"
    AREA = "Area"


class BuildingType(str, Enum):
    WALL = "Wall"
    DOOR = "Door"
    FURNITURE = "Furniture"
    PRODUCTION = "Production"
    STORAGE = "Storage"
    POWER = "Power"
    TEMPERATURE = "Temperature"
    SECURITY = "Security"
    MISC = "Misc"


class Position(BaseModel):
    x: int
    y: int
    z: int = 0

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def distance_to(self, other: 'Position') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Material(BaseModel):
    def_name: str
    stack_count: int = 1
    quality: Optional[str] = None
    hit_points: Optional[int] = None
    max_hit_points: Optional[int] = None


class Building(BaseModel):
    id: str
    def_name: str
    position: Position
    rotation: int = 0
    stuff_material: Optional[str] = None
    hit_points: Optional[int] = None
    max_hit_points: Optional[int] = None
    faction: Optional[str] = None
    building_type: Optional[BuildingType] = None
    is_blueprint: bool = False
    is_frame: bool = False
    work_left: Optional[float] = None
    room_id: Optional[str] = None
    power_consumption: Optional[float] = None
    is_powered: Optional[bool] = None
    custom_properties: Dict[str, Any] = Field(default_factory=dict)

    def is_complete(self) -> bool:
        return not self.is_blueprint and not self.is_frame


class Room(BaseModel):
    id: str
    positions: List[Position]
    role: Optional[str] = None
    impressiveness: Optional[float] = None
    wealth: Optional[float] = None
    space: Optional[float] = None
    beauty: Optional[float] = None
    cleanliness: Optional[float] = None
    outdoors: bool = False
    roof_coverage: float = 0.0
    temperature: Optional[float] = None
    connected_rooms: List[str] = Field(default_factory=list)


class Zone(BaseModel):
    id: str
    name: str
    zone_type: str
    cells: List[Position]
    color: Optional[str] = None
    allowed_items: List[str] = Field(default_factory=list)
    priority: int = 0


class TerrainTile(BaseModel):
    position: Position
    def_name: str
    affordances: List[str] = Field(default_factory=list)
    fertility: float = 1.0
    walkable: bool = True
    buildable: bool = True


class Pawn(BaseModel):
    id: str
    name: str
    position: Position
    faction: Optional[str] = None
    is_colonist: bool = False
    is_prisoner: bool = False
    is_guest: bool = False
    health: float = 1.0
    mood: Optional[float] = None
    skills: Dict[str, float] = Field(default_factory=dict)
    traits: List[str] = Field(default_factory=list)
    work_priorities: Dict[str, int] = Field(default_factory=dict)
    equipment: List[str] = Field(default_factory=list)
    inventory: List[Material] = Field(default_factory=list)
    bed_id: Optional[str] = None


class Item(BaseModel):
    id: str
    def_name: str
    position: Position
    stack_count: int = 1
    quality: Optional[str] = None
    hit_points: Optional[int] = None
    max_hit_points: Optional[int] = None
    rot_progress: Optional[float] = None
    forbidden: bool = False
    in_storage: bool = False


class Map(BaseModel):
    map_id: str
    size: Tuple[int, int]
    terrain: Dict[Tuple[int, int], TerrainTile] = Field(default_factory=dict)
    buildings: List[Building] = Field(default_factory=list)
    pawns: List[Pawn] = Field(default_factory=list)
    items: List[Item] = Field(default_factory=list)
    zones: List[Zone] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)
    home_area: List[Position] = Field(default_factory=list)
    roof_area: List[Position] = Field(default_factory=list)
    
    def get_building_at(self, position: Position) -> Optional[Building]:
        for building in self.buildings:
            if building.position == position:
                return building
        return None
    
    def get_terrain_at(self, x: int, y: int) -> Optional[TerrainTile]:
        return self.terrain.get((x, y))
    
    def get_colonists(self) -> List[Pawn]:
        return [p for p in self.pawns if p.is_colonist]
    
    def get_buildings_by_type(self, building_type: BuildingType) -> List[Building]:
        return [b for b in self.buildings if b.building_type == building_type]


class GameState(BaseModel):
    save_version: str
    game_version: str
    save_name: str
    seed: str
    play_time: float
    tick: int
    date: str
    maps: List[Map] = Field(default_factory=list)
    faction_relations: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    research: Dict[str, bool] = Field(default_factory=dict)
    resources: Dict[str, int] = Field(default_factory=dict)
    mod_ids: List[str] = Field(default_factory=list)
    mod_names: List[str] = Field(default_factory=list)
    difficulty: Optional[str] = None
    storyteller: Optional[str] = None
    
    def get_active_map(self) -> Optional[Map]:
        return self.maps[0] if self.maps else None
    
    def get_total_colonists(self) -> int:
        return sum(len(m.get_colonists()) for m in self.maps)
    
    def get_total_wealth(self) -> float:
        return sum(sum(r.wealth or 0 for r in m.rooms) for m in self.maps)
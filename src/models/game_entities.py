from typing import Any
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
    BRIDGE = "Bridge"
    FLOOR = "Floor"
    CONDUIT = "Conduit"
    FENCE = "Fence"
    LIGHT = "Light"
    MISC = "Misc"


class Position(BaseModel):
    x: int
    y: int
    z: int = 0

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Material(BaseModel):
    def_name: str
    stack_count: int = 1
    quality: str | None = None
    hit_points: int | None = None
    max_hit_points: int | None = None


class Building(BaseModel):
    id: str
    def_name: str
    position: Position
    rotation: int = 0
    stuff_material: str | None = None
    hit_points: int | None = None
    max_hit_points: int | None = None
    faction: str | None = None
    building_type: BuildingType | None = None
    is_blueprint: bool = False
    is_frame: bool = False
    work_left: float | None = None
    room_id: str | None = None
    power_consumption: float | None = None
    is_powered: bool | None = None
    custom_properties: dict[str, Any] = Field(default_factory=dict)

    def is_complete(self) -> bool:
        return not self.is_blueprint and not self.is_frame


class Room(BaseModel):
    id: str
    positions: list[Position]
    role: str | None = None
    impressiveness: float | None = None
    wealth: float | None = None
    space: float | None = None
    beauty: float | None = None
    cleanliness: float | None = None
    outdoors: bool = False
    roof_coverage: float = 0.0
    temperature: float | None = None
    connected_rooms: list[str] = Field(default_factory=list)


class Zone(BaseModel):
    id: str
    name: str
    zone_type: str
    cells: list[Position]
    color: str | None = None
    allowed_items: list[str] = Field(default_factory=list)
    priority: int = 0


class TerrainTile(BaseModel):
    position: Position
    def_name: str
    affordances: list[str] = Field(default_factory=list)
    fertility: float = 1.0
    walkable: bool = True
    buildable: bool = True


class Pawn(BaseModel):
    id: str
    name: str
    position: Position
    faction: str | None = None
    is_colonist: bool = False
    is_prisoner: bool = False
    is_guest: bool = False
    health: float = 1.0
    mood: float | None = None
    skills: dict[str, float] = Field(default_factory=dict)
    traits: list[str] = Field(default_factory=list)
    work_priorities: dict[str, int] = Field(default_factory=dict)
    equipment: list[str] = Field(default_factory=list)
    inventory: list[Material] = Field(default_factory=list)
    bed_id: str | None = None


class Item(BaseModel):
    id: str
    def_name: str
    position: Position
    stack_count: int = 1
    quality: str | None = None
    hit_points: int | None = None
    max_hit_points: int | None = None
    rot_progress: float | None = None
    forbidden: bool = False
    in_storage: bool = False


class Map(BaseModel):
    map_id: str
    size: tuple[int, int]
    terrain: dict[tuple[int, int], TerrainTile] = Field(default_factory=dict)
    buildings: list[Building] = Field(default_factory=list)
    pawns: list[Pawn] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    home_area: list[Position] = Field(default_factory=list)
    roof_area: list[Position] = Field(default_factory=list)

    def get_building_at(self, position: Position) -> Building | None:
        for building in self.buildings:
            if building.position == position:
                return building
        return None

    def get_terrain_at(self, x: int, y: int) -> TerrainTile | None:
        return self.terrain.get((x, y))

    def get_colonists(self) -> list[Pawn]:
        return [p for p in self.pawns if p.is_colonist]

    def get_buildings_by_type(self, building_type: BuildingType) -> list[Building]:
        return [b for b in self.buildings if b.building_type == building_type]


class GameState(BaseModel):
    save_version: str
    game_version: str
    save_name: str
    seed: str
    play_time: float
    tick: int
    date: str
    maps: list[Map] = Field(default_factory=list)
    faction_relations: dict[str, dict[str, float]] = Field(default_factory=dict)
    research: dict[str, bool] = Field(default_factory=dict)
    resources: dict[str, int] = Field(default_factory=dict)
    mod_ids: list[str] = Field(default_factory=list)
    mod_names: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    storyteller: str | None = None

    def get_active_map(self) -> Map | None:
        return self.maps[0] if self.maps else None

    def get_total_colonists(self) -> int:
        return sum(len(m.get_colonists()) for m in self.maps)

    def get_total_wealth(self) -> float:
        return sum(sum(r.wealth or 0 for r in m.rooms) for m in self.maps)

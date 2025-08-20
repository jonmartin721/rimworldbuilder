"""
Hybrid generator that combines real prefabs with procedural generation.
Uses actual AlphaPrefabs designs as anchor points and fills between them with WFC.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass

from src.generators.improved_wfc_generator import ImprovedWFCGenerator
from src.generators.alpha_prefab_parser import AlphaPrefabParser, AlphaPrefabLayout
from src.generators.wfc_generator import TileType


@dataclass
class PlacedPrefab:
    """Represents a prefab that has been placed on the grid"""
    layout: AlphaPrefabLayout
    position: Tuple[int, int]  # Top-left corner
    rotation: int  # 0, 90, 180, 270 degrees
    category: str


class HybridPrefabGenerator(ImprovedWFCGenerator):
    """Generator that uses real prefabs as anchors with WFC filling between them"""
    
    def __init__(self, width: int, height: int, 
                 alpha_prefabs_path: Optional[Path] = None,
                 learned_patterns_file: Optional[Path] = None):
        """
        Initialize hybrid generator.
        
        Args:
            width: Grid width
            height: Grid height  
            alpha_prefabs_path: Path to AlphaPrefabs mod
            learned_patterns_file: Path to learned patterns JSON
        """
        super().__init__(width, height, learned_patterns_file)
        
        self.placed_prefabs: List[PlacedPrefab] = []
        self.prefab_library: Dict[str, List[AlphaPrefabLayout]] = {}
        
        # Load prefab library if path provided
        if alpha_prefabs_path and alpha_prefabs_path.exists():
            self._load_prefab_library(alpha_prefabs_path)
    
    def _load_prefab_library(self, alpha_prefabs_path: Path):
        """Load and categorize all available prefabs"""
        parser = AlphaPrefabParser(alpha_prefabs_path)
        all_layouts = parser.parse_all_layouts()
        
        # Categorize prefabs
        for layout in all_layouts:
            category = self._categorize_prefab(layout.def_name)
            if category not in self.prefab_library:
                self.prefab_library[category] = []
            self.prefab_library[category].append(layout)
        
        print(f"Loaded {len(all_layouts)} prefabs in {len(self.prefab_library)} categories")
    
    def _categorize_prefab(self, def_name: str) -> str:
        """Categorize a prefab by its name"""
        name_lower = def_name.lower()
        
        if 'bedroom' in name_lower:
            return 'bedroom'
        elif 'kitchen' in name_lower or 'eating' in name_lower or 'dining' in name_lower:
            return 'kitchen'
        elif 'storage' in name_lower or 'warehouse' in name_lower:
            return 'storage'
        elif 'workshop' in name_lower or 'production' in name_lower:
            return 'workshop'
        elif 'hospital' in name_lower or 'medical' in name_lower:
            return 'medical'
        elif 'rec' in name_lower or 'joy' in name_lower:
            return 'recreation'
        elif 'power' in name_lower or 'battery' in name_lower:
            return 'power'
        elif 'research' in name_lower or 'lab' in name_lower:
            return 'research'
        elif 'prison' in name_lower:
            return 'prison'
        elif 'barn' in name_lower or 'animal' in name_lower:
            return 'animal'
        else:
            return 'general'
    
    def reset(self):
        """Reset the grid to empty state"""
        self.grid = np.full((self.height, self.width), TileType.EMPTY.value)
        self.collapsed = np.zeros((self.height, self.width), dtype=bool)
        self.placed_prefabs = []
        self.room_placements = []
    
    def generate_with_prefab_anchors(self,
                                    buildable_mask: Optional[np.ndarray] = None,
                                    prefab_categories: List[str] = None,
                                    num_prefabs: int = 3,
                                    fill_with_wfc: bool = True) -> np.ndarray:
        """
        Generate base using real prefabs as anchor points.
        
        Args:
            buildable_mask: Boolean array of buildable areas
            prefab_categories: Categories of prefabs to use (e.g., ['bedroom', 'kitchen'])
            num_prefabs: Number of prefab anchors to place
            fill_with_wfc: Whether to fill remaining space with WFC
            
        Returns:
            Generated grid with prefabs and WFC fill
        """
        # Reset grid
        self.reset()
        
        # Apply buildable mask
        if buildable_mask is not None:
            self._apply_buildable_mask(buildable_mask)
        
        # Default categories if not specified
        if prefab_categories is None:
            prefab_categories = ['bedroom', 'kitchen', 'storage', 'workshop']
        
        # Place prefab anchors
        placed_count = 0
        for category in prefab_categories:
            if category in self.prefab_library:
                prefabs = self.prefab_library[category]
                if prefabs:
                    # Select a suitable prefab
                    prefab = self._select_prefab(prefabs, buildable_mask)
                    if prefab:
                        if self._place_prefab(prefab, category):
                            placed_count += 1
                            if placed_count >= num_prefabs:
                                break
        
        print(f"Placed {placed_count} prefab anchors")
        
        # Fill remaining space with WFC if requested
        if fill_with_wfc:
            self._fill_with_wfc()
        
        # Connect prefabs with corridors
        self._connect_prefabs()
        
        return self.grid
    
    def _select_prefab(self, prefabs: List[AlphaPrefabLayout], 
                      buildable_mask: Optional[np.ndarray]) -> Optional[AlphaPrefabLayout]:
        """Select a suitable prefab based on available space"""
        # Filter by size - prefer smaller prefabs for easier placement
        suitable = []
        for prefab in prefabs:
            if prefab.width <= self.width // 3 and prefab.height <= self.height // 3:
                suitable.append(prefab)
        
        if not suitable:
            # Fall back to any prefab that fits
            for prefab in prefabs:
                if prefab.width <= self.width - 2 and prefab.height <= self.height - 2:
                    suitable.append(prefab)
        
        return random.choice(suitable) if suitable else None
    
    def _place_prefab(self, prefab: AlphaPrefabLayout, category: str) -> bool:
        """Place a real prefab on the grid"""
        # Find a suitable position
        position = self._find_prefab_position(prefab)
        if position is None:
            return False
        
        x, y = position
        
        # Place the prefab layout
        for py, row in enumerate(prefab.layout_grid):
            for px, item in enumerate(row):
                grid_x = x + px
                grid_y = y + py
                
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    # Map prefab item to our tile type
                    tile_type = self._map_item_to_tile(item)
                    self.grid[grid_y, grid_x] = tile_type.value
                    self.collapsed[grid_y, grid_x] = True
        
        # Track placement
        self.placed_prefabs.append(PlacedPrefab(
            layout=prefab,
            position=position,
            rotation=0,
            category=category
        ))
        
        print(f"  Placed {prefab.def_name} ({category}) at {position}")
        return True
    
    def _find_prefab_position(self, prefab: AlphaPrefabLayout) -> Optional[Tuple[int, int]]:
        """Find a valid position for the prefab"""
        # Try random positions
        for _ in range(50):
            x = random.randint(1, self.width - prefab.width - 1)
            y = random.randint(1, self.height - prefab.height - 1)
            
            if self._is_valid_prefab_position(x, y, prefab.width, prefab.height):
                return (x, y)
        
        # Systematic search if random fails
        for y in range(1, self.height - prefab.height - 1, 3):
            for x in range(1, self.width - prefab.width - 1, 3):
                if self._is_valid_prefab_position(x, y, prefab.width, prefab.height):
                    return (x, y)
        
        return None
    
    def _is_valid_prefab_position(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if prefab can be placed at position"""
        # Check bounds
        if x < 0 or y < 0 or x + width >= self.width or y + height >= self.height:
            return False
        
        # Check for overlaps with buffer
        for dy in range(-1, height + 1):
            for dx in range(-1, width + 1):
                px, py = x + dx, y + dy
                if 0 <= px < self.width and 0 <= py < self.height:
                    if self.collapsed[py, px]:
                        return False
        
        return True
    
    def _map_item_to_tile(self, item: str) -> TileType:
        """Map a prefab item string to our TileType"""
        if item == '.':
            return TileType.EMPTY
        elif 'Wall' in item:
            return TileType.WALL
        elif 'Door' in item:
            return TileType.DOOR
        elif 'Bed' in item or 'Bedroll' in item:
            return TileType.BEDROOM
        elif 'Table' in item or 'Chair' in item or 'Stool' in item:
            return TileType.KITCHEN
        elif 'Stove' in item or 'Kitchen' in item:
            return TileType.KITCHEN
        elif 'Storage' in item or 'Shelf' in item:
            return TileType.STORAGE
        elif 'Bench' in item or 'Production' in item:
            return TileType.WORKSHOP
        elif 'Medical' in item or 'Hospital' in item:
            return TileType.MEDICAL
        elif 'Research' in item:
            return TileType.RESEARCH
        elif 'Joy' in item or 'Recreation' in item:
            return TileType.RECREATION
        elif 'Power' in item or 'Battery' in item:
            return TileType.POWER
        else:
            # Default to wall for unknown items
            return TileType.WALL
    
    def _fill_with_wfc(self):
        """Fill remaining space using WFC"""
        # Run WFC on uncollapsed tiles
        max_iterations = 1000
        iterations = 0
        
        while iterations < max_iterations:
            # Find lowest entropy uncollapsed tile
            min_entropy = float('inf')
            best_tile = None
            
            for y in range(self.height):
                for x in range(self.width):
                    if not self.collapsed[y, x]:
                        # Simple entropy: distance to nearest prefab
                        entropy = self._distance_to_nearest_prefab(x, y)
                        if entropy < min_entropy:
                            min_entropy = entropy
                            best_tile = (x, y)
            
            if best_tile is None:
                break
            
            # Collapse this tile
            x, y = best_tile
            tile_type = self._choose_tile_type_near_prefabs(x, y)
            self.grid[y, x] = tile_type.value
            self.collapsed[y, x] = True
            
            iterations += 1
        
        print(f"  WFC filled {iterations} tiles")
    
    def _distance_to_nearest_prefab(self, x: int, y: int) -> float:
        """Calculate distance to nearest placed prefab"""
        min_dist = float('inf')
        
        for placed in self.placed_prefabs:
            px, py = placed.position
            pw, ph = placed.layout.width, placed.layout.height
            
            # Distance to prefab center
            cx = px + pw // 2
            cy = py + ph // 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _choose_tile_type_near_prefabs(self, x: int, y: int) -> TileType:
        """Choose appropriate tile type based on nearby prefabs"""
        # Check adjacent tiles
        adjacent_types = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.collapsed[ny, nx]:
                    adjacent_types.append(TileType(self.grid[ny, nx]))
        
        # If next to rooms, likely a corridor or wall
        if adjacent_types:
            if TileType.CORRIDOR in adjacent_types:
                return TileType.CORRIDOR
            elif any(t in [TileType.BEDROOM, TileType.KITCHEN, TileType.STORAGE] for t in adjacent_types):
                return TileType.WALL
        
        # Default to empty
        return TileType.EMPTY
    
    def _connect_prefabs(self):
        """Connect placed prefabs with corridors"""
        # Connect each prefab to its nearest neighbor
        for i, prefab1 in enumerate(self.placed_prefabs):
            if i < len(self.placed_prefabs) - 1:
                prefab2 = self.placed_prefabs[i + 1]
                self._create_corridor_between_prefabs(prefab1, prefab2)
    
    def _create_corridor_between_prefabs(self, prefab1: PlacedPrefab, prefab2: PlacedPrefab):
        """Create corridor between two prefabs"""
        # Get center points
        x1 = prefab1.position[0] + prefab1.layout.width // 2
        y1 = prefab1.position[1] + prefab1.layout.height // 2
        x2 = prefab2.position[0] + prefab2.layout.width // 2
        y2 = prefab2.position[1] + prefab2.layout.height // 2
        
        # Create L-shaped corridor
        if random.random() < 0.5:
            # Horizontal first
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < self.width and 0 <= y1 < self.height:
                    if self.grid[y1, x] == TileType.EMPTY.value:
                        self.grid[y1, x] = TileType.CORRIDOR.value
            # Then vertical
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x2 < self.width and 0 <= y < self.height:
                    if self.grid[y, x2] == TileType.EMPTY.value:
                        self.grid[y, x2] = TileType.CORRIDOR.value
        else:
            # Vertical first
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x1 < self.width and 0 <= y < self.height:
                    if self.grid[y, x1] == TileType.EMPTY.value:
                        self.grid[y, x1] = TileType.CORRIDOR.value
            # Then horizontal
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= x < self.width and 0 <= y2 < self.height:
                    if self.grid[y2, x] == TileType.EMPTY.value:
                        self.grid[y2, x] = TileType.CORRIDOR.value
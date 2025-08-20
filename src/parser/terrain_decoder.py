"""
RimWorld terrain grid decoder.
Based on community reverse engineering efforts.
"""

import base64
import zlib
import struct
from typing import Dict, List, Tuple, Optional
import numpy as np
from lxml import etree
import logging

logger = logging.getLogger(__name__)


class TerrainDecoder:
    """Decodes RimWorld's compressed terrain grid data"""
    
    # Known terrain type IDs (16-bit values)
    # These are community-discovered mappings
    TERRAIN_IDS = {
        0xA790: "WaterOceanDeep",      # 42896 / -22640 signed
        0x86A1: "Soil",                # 34465 / -31071 signed  
        0x8606: "MarshyTerrain",       # 34310
        0x2CB5: "WaterShallow",        # 11445
        0x798C: "WaterDeep",           # 31116
        0x52A6: "Sand",                # 21158
        0x4546: "Gravel",              # 17734
        0x89FF: "SoilRich",            # 35327 / -30209 signed
        # Bridge types would have their own IDs
        # We'll discover them from the data
    }
    
    # Foundation IDs (buildable artificial terrain)
    FOUNDATION_IDS = {
        0x0000: "None",
        0x1A47: "LightBridge",  # Wood/light bridge foundation (437 tiles)
        0x8C7D: "HeavyBridge",  # Heavy/stone bridge foundation (6924 tiles)
    }
    
    def __init__(self):
        self.terrain_lookup = {}
        self.unknown_ids = set()
        self.foundation_grid = None
    
    def decode_terrain_grid(self, map_elem: etree.Element) -> Optional[np.ndarray]:
        """
        Decode the terrain grid from a map element.
        
        Args:
            map_elem: The map XML element from save file
            
        Returns:
            2D numpy array with terrain IDs, or None if failed
        """
        # Get map dimensions
        map_width, map_height = self._get_map_size(map_elem)
        logger.info(f"Map size: {map_width}x{map_height}")
        
        # Find terrain grid
        terrain_grid = map_elem.find('terrainGrid')
        if terrain_grid is None:
            logger.error("No terrainGrid found")
            return None
        
        # Decode topGridDeflate (surface terrain)
        top_grid = terrain_grid.find('topGridDeflate')
        if top_grid is None or not top_grid.text:
            logger.error("No topGridDeflate found")
            return None
        
        try:
            # Decode base64
            compressed = base64.b64decode(top_grid.text.strip())
            
            # Decompress using raw deflate (no headers)
            decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
            
            # Parse as 16-bit unsigned integers (2 bytes per tile)
            expected_size = map_width * map_height * 2
            if len(decompressed) != expected_size:
                logger.warning(f"Decompressed size {len(decompressed)} != expected {expected_size}")
            
            # Unpack as unsigned 16-bit integers (little-endian)
            num_tiles = len(decompressed) // 2
            terrain_ids = struct.unpack(f'<{num_tiles}H', decompressed[:num_tiles * 2])
            
            # Convert to 2D array
            grid = np.array(terrain_ids, dtype=np.uint16)
            grid = grid.reshape((map_height, map_width))
            
            # Analyze terrain types
            self._analyze_terrain_types(grid)
            
            return grid
            
        except Exception as e:
            logger.error(f"Failed to decode terrain grid: {e}")
            return None
    
    def decode_foundation_grid(self, map_elem: etree.Element) -> Optional[np.ndarray]:
        """Decode the foundationGridDeflate (buildable bridges/floors)"""
        terrain_grid = map_elem.find('terrainGrid')
        if terrain_grid is None:
            return None
        
        foundation_elem = terrain_grid.find('foundationGridDeflate')
        if foundation_elem is None or not foundation_elem.text:
            return None
        
        try:
            compressed = base64.b64decode(foundation_elem.text.strip())
            decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
            
            map_width, map_height = self._get_map_size(map_elem)
            num_tiles = len(decompressed) // 2
            foundation_ids = struct.unpack(f'<{num_tiles}H', decompressed[:num_tiles * 2])
            
            grid = np.array(foundation_ids, dtype=np.uint16)
            grid = grid.reshape((map_height, map_width))
            
            self.foundation_grid = grid
            
            # Log foundation statistics
            unique, counts = np.unique(grid, return_counts=True)
            logger.info("Foundation grid decoded:")
            for val, count in zip(unique, counts):
                name = self.FOUNDATION_IDS.get(int(val), f"Unknown_{val:04X}")
                logger.info(f"  {name}: {count} tiles")
            
            return grid
            
        except Exception as e:
            logger.error(f"Failed to decode foundation grid: {e}")
            return None
    
    def decode_under_grid(self, map_elem: etree.Element) -> Optional[np.ndarray]:
        """Decode the underGridDeflate (terrain under buildings)"""
        terrain_grid = map_elem.find('terrainGrid')
        if terrain_grid is None:
            return None
        
        under_grid = terrain_grid.find('underGridDeflate')
        if under_grid is None or not under_grid.text:
            return None
        
        try:
            # Same process as topGrid
            compressed = base64.b64decode(under_grid.text.strip())
            decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
            
            map_width, map_height = self._get_map_size(map_elem)
            num_tiles = len(decompressed) // 2
            terrain_ids = struct.unpack(f'<{num_tiles}H', decompressed[:num_tiles * 2])
            
            grid = np.array(terrain_ids, dtype=np.uint16)
            grid = grid.reshape((map_height, map_width))
            
            return grid
            
        except Exception as e:
            logger.error(f"Failed to decode under grid: {e}")
            return None
    
    def _get_map_size(self, map_elem: etree.Element) -> Tuple[int, int]:
        """Extract map dimensions from map element"""
        # First try to detect from decompressed data size
        terrain_grid = map_elem.find('terrainGrid')
        if terrain_grid is not None:
            top_grid = terrain_grid.find('topGridDeflate')
            if top_grid is not None and top_grid.text:
                try:
                    compressed = base64.b64decode(top_grid.text.strip())
                    decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
                    num_tiles = len(decompressed) // 2
                    
                    # Check common map sizes
                    for size in [275, 250, 300, 225, 200]:
                        if num_tiles == size * size:
                            return size, size
                except:
                    pass
        
        # Fall back to mapInfo
        map_info = map_elem.find('.//mapInfo')
        if map_info is not None:
            size = map_info.find('.//size')
            if size is not None:
                x = size.find('x')
                z = size.find('z')
                if x is not None and z is not None and x.text and z.text:
                    return int(x.text), int(z.text)
        
        return 250, 250  # Default RimWorld map size
    
    def _analyze_terrain_types(self, grid: np.ndarray):
        """Analyze and log terrain types found in grid"""
        unique_ids = np.unique(grid)
        logger.info(f"Found {len(unique_ids)} unique terrain types")
        
        # Count occurrences
        terrain_counts = {}
        for terrain_id in unique_ids:
            count = np.sum(grid == terrain_id)
            terrain_name = self.TERRAIN_IDS.get(int(terrain_id), f"Unknown_{terrain_id:04X}")
            terrain_counts[terrain_name] = count
            
            if terrain_id not in self.TERRAIN_IDS:
                self.unknown_ids.add(terrain_id)
        
        # Log most common terrains
        sorted_terrains = sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("Most common terrain types:")
        for name, count in sorted_terrains[:10]:
            logger.info(f"  {name}: {count} tiles")
        
        # Check for bridges
        self._check_for_bridges(grid, unique_ids)
    
    def _check_for_bridges(self, grid: np.ndarray, unique_ids: np.ndarray):
        """Check if any terrain IDs might be bridges"""
        # Bridge terrain IDs are not well documented
        # Look for IDs that might be bridges based on patterns
        
        possible_bridge_ids = []
        
        for terrain_id in unique_ids:
            # Bridges are typically less common than base terrain
            count = np.sum(grid == terrain_id)
            
            # Heuristic: bridges are uncommon but not rare
            # Usually between 10-1000 tiles on a map
            if 10 <= count <= 1000:
                # Check if this ID is unknown
                if terrain_id not in self.TERRAIN_IDS:
                    possible_bridge_ids.append((terrain_id, count))
        
        if possible_bridge_ids:
            logger.info("Possible bridge terrain IDs (based on occurrence patterns):")
            for terrain_id, count in possible_bridge_ids:
                logger.info(f"  ID 0x{terrain_id:04X} ({terrain_id}): {count} tiles")
    
    def find_bridges(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find positions of bridge tiles in the terrain grid.
        
        Returns:
            List of (x, y) positions where bridges are located
        """
        bridge_positions = []
        
        # Look for specific bridge IDs if we know them
        # These would need to be discovered from actual save files
        BRIDGE_IDS = [
            # Add discovered bridge IDs here
            # For example: 0x1234, 0x5678
        ]
        
        for bridge_id in BRIDGE_IDS:
            positions = np.argwhere(grid == bridge_id)
            for y, x in positions:
                bridge_positions.append((x, y))
        
        # Also check unknown IDs that might be bridges
        for terrain_id in self.unknown_ids:
            count = np.sum(grid == terrain_id)
            if 10 <= count <= 1000:  # Heuristic for bridge-like patterns
                positions = np.argwhere(grid == terrain_id)
                # Could add additional checks here
                # e.g., bridges tend to be in lines or clusters
        
        return bridge_positions
    
    def get_terrain_name(self, terrain_id: int) -> str:
        """Get human-readable name for terrain ID"""
        return self.TERRAIN_IDS.get(terrain_id, f"Unknown_{terrain_id:04X}")


# Standalone test function
def test_terrain_decoder(save_path: str = 'data/saves/Autosave-2.rws'):
    """Test the terrain decoder on a save file"""
    from lxml import etree
    
    print(f"Testing terrain decoder on {save_path}")
    
    tree = etree.parse(save_path)
    root = tree.getroot()
    game = root.find('game')
    maps = game.find('maps')
    first_map = maps.find('li')
    
    decoder = TerrainDecoder()
    
    # Decode main terrain
    terrain_grid = decoder.decode_terrain_grid(first_map)
    if terrain_grid is not None:
        print(f"Successfully decoded terrain grid: {terrain_grid.shape}")
        
        # Look for specific patterns
        unique_vals = np.unique(terrain_grid)
        print(f"\nUnique terrain IDs found: {len(unique_vals)}")
        
        # Show unknown IDs that might be bridges
        if decoder.unknown_ids:
            print(f"\nUnknown terrain IDs (possible bridges):")
            for tid in sorted(decoder.unknown_ids):
                count = np.sum(terrain_grid == tid)
                print(f"  0x{tid:04X} ({tid}): {count} tiles")
    
    # Decode under grid
    under_grid = decoder.decode_under_grid(first_map)
    if under_grid is not None:
        print(f"\nSuccessfully decoded under grid: {under_grid.shape}")
        
        # Check if under grid has different values
        under_unique = np.unique(under_grid)
        print(f"Unique under-terrain IDs: {len(under_unique)}")


if __name__ == "__main__":
    test_terrain_decoder()
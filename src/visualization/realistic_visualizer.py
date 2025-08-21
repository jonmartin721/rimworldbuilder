"""
Visualizer for realistic RimWorld base layouts with proper furniture rendering.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from src.generators.realistic_base_generator import CellType


class RealisticBaseVisualizer:
    """Visualizer for realistic base layouts with furniture"""
    
    # Rich color scheme for different cell types
    CELL_COLORS = {
        CellType.EMPTY: (30, 30, 30),           # Dark background
        CellType.WALL: (120, 120, 120),         # Gray stone wall
        CellType.DOOR: (139, 90, 43),           # Brown wood door
        CellType.BED: (150, 75, 0),             # Dark orange bed
        CellType.TABLE: (160, 82, 45),          # Sienna table
        CellType.CHAIR: (205, 133, 63),         # Peru chair
        CellType.STORAGE: (255, 215, 0),        # Gold storage
        CellType.WORKBENCH: (70, 130, 180),     # Steel blue workbench
        CellType.STOVE: (255, 140, 0),          # Dark orange stove
        CellType.TORCH: (255, 223, 0),          # Bright yellow torch
        CellType.FLOOR: (101, 67, 33),          # Dark brown floor
        CellType.DRESSER: (139, 69, 19),        # Saddle brown dresser
        CellType.ENDTABLE: (160, 82, 45),       # Sienna end table
        CellType.RECREATION: (218, 112, 214),   # Orchid recreation
        CellType.MEDICAL_BED: (255, 99, 71),    # Tomato medical bed
        CellType.RESEARCH_BENCH: (147, 112, 219), # Medium purple research
        CellType.BATTERY: (50, 205, 50),        # Lime green battery
        CellType.SOLAR_PANEL: (135, 206, 235),  # Sky blue solar
        CellType.WIND_TURBINE: (176, 224, 230), # Powder blue turbine
        CellType.COOLER: (173, 216, 230),       # Light blue cooler
        CellType.HEATER: (255, 69, 0),          # Red orange heater
        CellType.PLANT_POT: (34, 139, 34),      # Forest green plant
        CellType.SANDBAG: (194, 178, 128),      # Sand color
        CellType.TURRET: (105, 105, 105),       # Dim gray turret
    }
    
    # Symbols for ASCII representation
    CELL_SYMBOLS = {
        CellType.EMPTY: ' ',
        CellType.WALL: '#',
        CellType.DOOR: '+',
        CellType.BED: 'B',
        CellType.TABLE: 'T',
        CellType.CHAIR: 'c',
        CellType.STORAGE: 'S',
        CellType.WORKBENCH: 'W',
        CellType.STOVE: 'K',
        CellType.TORCH: '*',
        CellType.FLOOR: '.',
        CellType.DRESSER: 'D',
        CellType.ENDTABLE: 'e',
        CellType.RECREATION: 'R',
        CellType.MEDICAL_BED: 'M',
        CellType.RESEARCH_BENCH: 'L',
        CellType.BATTERY: 'b',
        CellType.SOLAR_PANEL: 'O',
        CellType.WIND_TURBINE: 'Y',
        CellType.COOLER: '-',
        CellType.HEATER: '=',
        CellType.PLANT_POT: 'p',
        CellType.SANDBAG: 's',
        CellType.TURRET: 'X',
    }
    
    def __init__(self, scale: int = 10):
        """
        Initialize visualizer.
        
        Args:
            scale: Pixels per grid cell
        """
        self.scale = scale
        
    def visualize(self, grid: np.ndarray, output_path: str, 
                  title: str = "RimWorld Base Layout") -> None:
        """
        Create a visual representation of the base.
        
        Args:
            grid: 2D numpy array with cell types
            output_path: Path to save the image
            title: Title for the visualization
        """
        height, width = grid.shape
        
        # Create image
        img_width = width * self.scale
        img_height = height * self.scale + 60  # Extra space for title and legend
        
        image = Image.new('RGB', (img_width, img_height), color=(20, 20, 20))
        draw = ImageDraw.Draw(image)
        
        # Draw title
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        draw.text((10, 5), title, fill=(255, 255, 255), font=font)
        
        # Draw grid cells
        for y in range(height):
            for x in range(width):
                cell_value = int(grid[y, x])
                cell_type = CellType(cell_value)
                color = self.CELL_COLORS.get(cell_type, (100, 100, 100))
                
                # Calculate pixel coordinates
                px = x * self.scale
                py = y * self.scale + 30  # Offset for title
                
                # Draw cell
                draw.rectangle(
                    [px, py, px + self.scale - 1, py + self.scale - 1],
                    fill=color
                )
                
                # Add border for walls
                if cell_type == CellType.WALL:
                    draw.rectangle(
                        [px, py, px + self.scale - 1, py + self.scale - 1],
                        outline=(80, 80, 80),
                        width=1
                    )
                
                # Add symbols for furniture
                if cell_type in [CellType.BED, CellType.TABLE, CellType.CHAIR, 
                                CellType.WORKBENCH, CellType.STOVE]:
                    symbol = self.CELL_SYMBOLS[cell_type]
                    draw.text(
                        (px + self.scale // 3, py + self.scale // 4),
                        symbol,
                        fill=(255, 255, 255),
                        font=font
                    )
        
        # Draw legend at bottom
        legend_y = img_height - 25
        legend_items = [
            ("Wall", CellType.WALL),
            ("Door", CellType.DOOR),
            ("Bed", CellType.BED),
            ("Table", CellType.TABLE),
            ("Storage", CellType.STORAGE),
            ("Floor", CellType.FLOOR),
        ]
        
        legend_x = 10
        for name, cell_type in legend_items:
            color = self.CELL_COLORS[cell_type]
            # Draw color box
            draw.rectangle(
                [legend_x, legend_y, legend_x + 15, legend_y + 15],
                fill=color
            )
            # Draw label
            draw.text((legend_x + 20, legend_y), name, fill=(200, 200, 200))
            legend_x += 80
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image.save(output_path)
        
    def visualize_ascii(self, grid: np.ndarray) -> str:
        """
        Create ASCII representation of the base.
        
        Args:
            grid: 2D numpy array with cell types
            
        Returns:
            ASCII string representation
        """
        height, width = grid.shape
        lines = []
        
        for y in range(height):
            line = ""
            for x in range(width):
                cell_value = int(grid[y, x])
                cell_type = CellType(cell_value)
                symbol = self.CELL_SYMBOLS.get(cell_type, '?')
                line += symbol
            lines.append(line)
            
        return "\n".join(lines)
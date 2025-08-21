"""
Base visualization module with proper color mapping and error handling.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


class BaseVisualizer:
    """Visualizer for generated base grids"""

    # Default color scheme
    DEFAULT_COLORS = {
        0: (50, 50, 50),  # Empty - dark gray
        1: (180, 180, 180),  # Wall - light gray
        2: (139, 90, 43),  # Floor - brown
        3: (160, 130, 90),  # Room - tan
        4: (101, 67, 33),  # Door - dark brown
        5: (70, 130, 180),  # Furniture - steel blue
        6: (255, 215, 0),  # Storage - gold
        7: (50, 205, 50),  # Power - green
        8: (255, 140, 0),  # Kitchen - orange
        9: (100, 150, 200),  # Bedroom - light blue
        10: (255, 107, 107),  # Medical - light red
        11: (147, 112, 219),  # Research - purple
        12: (218, 112, 214),  # Recreation - pink
        13: (100, 100, 100),  # Corridor - gray
        14: (255, 255, 0),  # Light/Lamp - yellow
        15: (165, 42, 42),  # Production - brown
    }

    # Tile type names for legend
    TILE_NAMES = {
        0: "Empty",
        1: "Wall",
        2: "Floor",
        3: "Room",
        4: "Door",
        5: "Furniture",
        6: "Storage",
        7: "Power",
        8: "Kitchen",
        9: "Bedroom",
        10: "Medical",
        11: "Research",
        12: "Recreation",
        13: "Corridor",
        14: "Light",
        15: "Production",
    }

    def __init__(self, scale: int = 8, show_grid: bool = True):
        """
        Initialize visualizer.

        Args:
            scale: Pixels per grid cell
            show_grid: Whether to show grid lines
        """
        self.scale = scale
        self.show_grid = show_grid
        self.colors = self.DEFAULT_COLORS.copy()

    def visualize(
        self,
        grid: np.ndarray,
        filename: str = "base_visualization.png",
        title: str | None = None,
        show_legend: bool = True,
    ) -> Image.Image:
        """
        Create visualization of base grid.

        Args:
            grid: 2D numpy array of tile types
            filename: Output filename
            title: Optional title for the image
            show_legend: Whether to show tile type legend

        Returns:
            PIL Image object
        """
        height, width = grid.shape

        # Calculate image dimensions
        img_width = width * self.scale
        img_height = height * self.scale

        # Add space for legend if needed
        if show_legend:
            legend_width = 150
            img_width += legend_width

        # Add space for title if needed
        if title:
            title_height = 30
            img_height += title_height
        else:
            title_height = 0

        # Create image with background
        img = Image.new("RGB", (img_width, img_height), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)

        # Draw title if provided
        if title:
            try:
                # Try to use a nice font
                font = ImageFont.truetype("arial.ttf", 16)
            except OSError:
                font = None

            draw.text(
                (img_width // 2, 15),
                title,
                fill=(255, 255, 255),
                font=font,
                anchor="mm",
            )

        # Offset for drawing grid
        y_offset = title_height

        # Draw grid cells
        unique_tiles = set()
        for y in range(height):
            for x in range(width):
                tile_value = int(grid[y, x])
                unique_tiles.add(tile_value)

                # Get color (with fallback)
                color = self.colors.get(tile_value, (128, 128, 128))

                # Calculate pixel coordinates
                px1 = x * self.scale
                py1 = y * self.scale + y_offset
                px2 = (x + 1) * self.scale - 1
                py2 = (y + 1) * self.scale - 1 + y_offset

                # Draw cell
                draw.rectangle([px1, py1, px2, py2], fill=color)

        # Draw grid lines if enabled
        if self.show_grid and self.scale > 4:
            grid_color = (60, 60, 60)

            # Vertical lines
            for x in range(width + 1):
                px = x * self.scale
                draw.line(
                    [(px, y_offset), (px, height * self.scale + y_offset)],
                    fill=grid_color,
                    width=1,
                )

            # Horizontal lines
            for y in range(height + 1):
                py = y * self.scale + y_offset
                draw.line([(0, py), (width * self.scale, py)], fill=grid_color, width=1)

        # Draw legend if enabled
        if show_legend:
            self._draw_legend(draw, width * self.scale, y_offset, unique_tiles)

        # Log statistics
        logger.info(f"Visualization created: {width}x{height} grid")
        logger.info(f"Unique tile types: {sorted(unique_tiles)}")

        # Save image
        img.save(filename)
        logger.info(f"Saved visualization to {filename}")

        return img

    def _draw_legend(
        self, draw: ImageDraw.Draw, x_start: int, y_start: int, tile_types: set
    ):
        """Draw legend showing tile types"""
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except OSError:
            font = None

        # Legend background
        draw.rectangle(
            [x_start, y_start, x_start + 150, y_start + len(tile_types) * 20 + 20],
            fill=(30, 30, 30),
        )

        # Title
        draw.text(
            (x_start + 10, y_start + 5), "Tile Types", fill=(255, 255, 255), font=font
        )

        # Draw each tile type
        y = y_start + 25
        for tile_type in sorted(tile_types):
            if tile_type in self.colors:
                color = self.colors[tile_type]
                name = self.TILE_NAMES.get(tile_type, f"Type {tile_type}")

                # Color box
                draw.rectangle(
                    [x_start + 10, y, x_start + 25, y + 12],
                    fill=color,
                    outline=(255, 255, 255),
                )

                # Label
                draw.text((x_start + 30, y + 2), name, fill=(200, 200, 200), font=font)

                y += 18

    def create_comparison(
        self, grids: dict[str, np.ndarray], filename: str = "comparison.png"
    ) -> Image.Image:
        """
        Create side-by-side comparison of multiple grids.

        Args:
            grids: Dictionary of {label: grid}
            filename: Output filename

        Returns:
            Combined PIL Image
        """
        if not grids:
            raise ValueError("No grids provided for comparison")

        # Create individual visualizations
        images = []
        max_height = 0
        total_width = 0

        for label, grid in grids.items():
            # Create temp image
            temp_img = self.visualize(
                grid, f"temp_{label}.png", title=label, show_legend=False
            )
            images.append(temp_img)

            # Track dimensions
            max_height = max(max_height, temp_img.height)
            total_width += temp_img.width + 10  # 10px spacing

        # Create combined image
        combined = Image.new("RGB", (total_width, max_height), color=(40, 40, 40))

        # Paste individual images
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width + 10

        # Save
        combined.save(filename)
        logger.info(f"Saved comparison to {filename}")

        return combined

    def analyze_grid(self, grid: np.ndarray) -> dict:
        """
        Analyze grid and return statistics.

        Args:
            grid: 2D numpy array

        Returns:
            Dictionary of statistics
        """
        unique, counts = np.unique(grid, return_counts=True)
        total_cells = grid.size

        stats = {
            "dimensions": grid.shape,
            "total_cells": total_cells,
            "tile_distribution": {},
        }

        for tile_type, count in zip(unique, counts):
            tile_name = self.TILE_NAMES.get(int(tile_type), f"Unknown_{tile_type}")
            percentage = (count / total_cells) * 100
            stats["tile_distribution"][tile_name] = {
                "count": int(count),
                "percentage": f"{percentage:.1f}%",
            }

        # Calculate metrics
        empty_cells = np.sum(grid == 0)
        wall_cells = np.sum(grid == 1)
        usable_cells = total_cells - empty_cells - wall_cells

        stats["metrics"] = {
            "empty_space": f"{(empty_cells / total_cells) * 100:.1f}%",
            "walls": f"{(wall_cells / total_cells) * 100:.1f}%",
            "usable_space": f"{(usable_cells / total_cells) * 100:.1f}%",
            "density": f"{((total_cells - empty_cells) / total_cells) * 100:.1f}%",
        }

        return stats


def quick_visualize(grid: np.ndarray, filename: str = "quick_vis.png"):
    """Quick visualization function for convenience"""
    visualizer = BaseVisualizer()
    return visualizer.visualize(grid, filename)

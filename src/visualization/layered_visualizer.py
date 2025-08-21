"""
Enhanced layered visualization system with individual layer controls.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


class LayeredVisualizer:
    """Visualizer with layer support for better analysis"""

    # Tile type string to integer mapping
    TILE_TYPE_MAP = {
        "empty": 0,
        "wall": 1,
        "door": 2,
        "bedroom": 3,
        "storage": 4,
        "workshop": 5,
        "kitchen": 6,
        "dining": 7,
        "recreation": 8,
        "hospital": 9,
        "research": 10,
        "power": 11,
        "battery": 12,
        "corridor": 13,
        "outdoor": 14,
        "farm": 15,
    }

    # Layer definitions
    LAYERS = {
        "walls": {
            "tiles": [1],
            "color": (180, 180, 180),
            "name": "Walls",
            "enabled": True,
        },
        "doors": {
            "tiles": [2],
            "color": (101, 67, 33),
            "name": "Doors",
            "enabled": True,
        },
        "bedroom": {
            "tiles": [3],
            "color": (100, 150, 200),
            "name": "Bedrooms",
            "enabled": True,
        },
        "storage": {
            "tiles": [4],
            "color": (255, 215, 0),
            "name": "Storage",
            "enabled": True,
        },
        "workshop": {
            "tiles": [5],
            "color": (139, 90, 43),
            "name": "Workshop",
            "enabled": True,
        },
        "kitchen": {
            "tiles": [6],
            "color": (255, 140, 0),
            "name": "Kitchen",
            "enabled": True,
        },
        "dining": {
            "tiles": [7],
            "color": (200, 100, 50),
            "name": "Dining",
            "enabled": True,
        },
        "recreation": {
            "tiles": [8],
            "color": (218, 112, 214),
            "name": "Recreation",
            "enabled": True,
        },
        "medical": {
            "tiles": [9],
            "color": (255, 107, 107),
            "name": "Medical",
            "enabled": True,
        },
        "research": {
            "tiles": [10],
            "color": (147, 112, 219),
            "name": "Research",
            "enabled": True,
        },
        "power": {
            "tiles": [11, 12],
            "color": (50, 205, 50),
            "name": "Power & Battery",
            "enabled": True,
        },
        "corridor": {
            "tiles": [13],
            "color": (220, 220, 220),
            "name": "Corridors",
            "enabled": True,
        },
        "outdoor": {
            "tiles": [14],
            "color": (34, 139, 34),
            "name": "Outdoor",
            "enabled": True,
        },
        "farm": {
            "tiles": [15],
            "color": (165, 42, 42),
            "name": "Farm",
            "enabled": True,
        },
    }

    def __init__(self, scale: int = 8):
        """Initialize layered visualizer"""
        self.scale = scale
        self.enabled_layers = {k: v["enabled"] for k, v in self.LAYERS.items()}

    def set_layer_visibility(self, layer_name: str, visible: bool):
        """Toggle layer visibility"""
        if layer_name in self.enabled_layers:
            self.enabled_layers[layer_name] = visible

    def get_layer_info(self) -> list[dict]:
        """Get information about all layers"""
        return [
            {
                "id": k,
                "name": v["name"],
                "color": v["color"],
                "enabled": self.enabled_layers[k],
            }
            for k, v in self.LAYERS.items()
        ]

    def visualize(
        self,
        grid: np.ndarray,
        title: str | None = None,
        show_grid: bool = True,
        show_legend: bool = True,
        flip_y: bool = False,
    ) -> Image.Image:
        """
        Create layered visualization of base grid.

        Args:
            grid: 2D numpy array of tile types
            title: Optional title for the image
            show_grid: Whether to show grid lines
            show_legend: Whether to show layer legend
            flip_y: Whether to flip Y-axis (fix inverted display)

        Returns:
            PIL Image object
        """
        height, width = grid.shape

        # Calculate image dimensions
        img_width = width * self.scale
        img_height = height * self.scale

        # Add space for legend if needed
        legend_width = 180 if show_legend else 0
        title_height = 40 if title else 0

        total_width = img_width + legend_width
        total_height = img_height + title_height

        # Create image
        img = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Draw title
        if title:
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except OSError:
                font = None
            draw.text(
                (total_width // 2, 20),
                title,
                fill=(255, 255, 255),
                font=font,
                anchor="mm",
            )

        # Create base layer (empty/background)
        base_color = (50, 50, 50)

        # Draw grid cells
        for y in range(height):
            for x in range(width):
                # Handle vertical flip (flip over X-axis means Y coordinates are inverted)
                # Flipping over X-axis means top becomes bottom
                display_y = (height - 1 - y) if flip_y else y

                # Get tile value and convert to int if it's a string
                raw_value = grid[display_y, x]
                if isinstance(raw_value, str):
                    tile_value = self.TILE_TYPE_MAP.get(raw_value, 0)
                else:
                    tile_value = int(raw_value)

                # Find which layer this tile belongs to
                color = base_color
                for layer_id, layer_data in self.LAYERS.items():
                    if (
                        tile_value in layer_data["tiles"]
                        and self.enabled_layers[layer_id]
                    ):
                        color = layer_data["color"]
                        break

                # Calculate pixel coordinates
                px1 = x * self.scale
                py1 = y * self.scale + title_height
                px2 = (x + 1) * self.scale - 1
                py2 = (y + 1) * self.scale - 1 + title_height

                # Draw cell
                draw.rectangle([px1, py1, px2, py2], fill=color)

        # Draw grid lines if enabled
        if show_grid and self.scale > 4:
            grid_color = (70, 70, 70)

            # Vertical lines
            for x in range(width + 1):
                px = x * self.scale
                draw.line(
                    [(px, title_height), (px, height * self.scale + title_height)],
                    fill=grid_color,
                    width=1,
                )

            # Horizontal lines
            for y in range(height + 1):
                py = y * self.scale + title_height
                draw.line([(0, py), (width * self.scale, py)], fill=grid_color, width=1)

        # Draw legend if enabled
        if show_legend:
            self._draw_layer_legend(draw, img_width, title_height)

        return img

    def _draw_layer_legend(self, draw: ImageDraw.Draw, x_start: int, y_start: int):
        """Draw interactive layer legend"""
        try:
            font = ImageFont.truetype("arial.ttf", 11)
            small_font = ImageFont.truetype("arial.ttf", 9)
        except OSError:
            font = None
            small_font = None

        # Legend background
        legend_x = x_start + 10
        legend_y = y_start + 10
        legend_width = 160
        legend_height = len(self.LAYERS) * 25 + 40

        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
            fill=(40, 40, 40),
            outline=(100, 100, 100),
        )

        # Title
        draw.text(
            (legend_x + legend_width // 2, legend_y + 15),
            "Layers",
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )

        # Draw each layer
        y = legend_y + 35
        for layer_id, layer_data in self.LAYERS.items():
            enabled = self.enabled_layers[layer_id]

            # Checkbox
            check_x = legend_x + 10
            check_size = 12
            draw.rectangle(
                [check_x, y, check_x + check_size, y + check_size],
                fill=(100, 200, 100) if enabled else (60, 60, 60),
                outline=(150, 150, 150),
            )

            if enabled:
                # Draw checkmark
                draw.line(
                    [(check_x + 2, y + 6), (check_x + 5, y + 9)],
                    fill=(255, 255, 255),
                    width=2,
                )
                draw.line(
                    [(check_x + 5, y + 9), (check_x + 10, y + 3)],
                    fill=(255, 255, 255),
                    width=2,
                )

            # Color sample
            color_x = check_x + check_size + 8
            draw.rectangle(
                [color_x, y + 2, color_x + 20, y + check_size - 2],
                fill=layer_data["color"] if enabled else (60, 60, 60),
            )

            # Layer name
            text_x = color_x + 25
            text_color = (255, 255, 255) if enabled else (128, 128, 128)
            draw.text(
                (text_x, y + check_size // 2),
                layer_data["name"],
                fill=text_color,
                font=small_font,
                anchor="lm",
            )

            y += 25

    def create_layer_images(self, grid: np.ndarray) -> dict[str, Image.Image]:
        """Create separate image for each layer"""
        images = {}
        height, width = grid.shape

        for layer_id, layer_data in self.LAYERS.items():
            # Create image for this layer
            img = Image.new(
                "RGBA", (width * self.scale, height * self.scale), color=(0, 0, 0, 0)
            )
            draw = ImageDraw.Draw(img)

            # Draw only tiles belonging to this layer
            for y in range(height):
                for x in range(width):
                    tile_value = int(grid[y, x])

                    if tile_value in layer_data["tiles"]:
                        px1 = x * self.scale
                        py1 = y * self.scale
                        px2 = (x + 1) * self.scale - 1
                        py2 = (y + 1) * self.scale - 1

                        # Draw with transparency
                        color = layer_data["color"] + (255,)  # Add alpha
                        draw.rectangle([px1, py1, px2, py2], fill=color)

            images[layer_id] = img

        return images

    def create_composite(
        self, grid: np.ndarray, layers_to_show: list[str] | None = None
    ) -> Image.Image:
        """Create composite image with specified layers"""
        if layers_to_show is None:
            layers_to_show = [k for k, v in self.enabled_layers.items() if v]

        height, width = grid.shape

        # Create base image
        img = Image.new(
            "RGB", (width * self.scale, height * self.scale), color=(50, 50, 50)
        )

        # Get individual layer images
        layer_images = self.create_layer_images(grid)

        # Composite layers in order
        for layer_id in layers_to_show:
            if layer_id in layer_images:
                img.paste(layer_images[layer_id], (0, 0), layer_images[layer_id])

        return img

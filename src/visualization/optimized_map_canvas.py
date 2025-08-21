"""
Optimized map canvas with pre-rendered layers and efficient rendering
"""

import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import threading
from dataclasses import dataclass


class ModernTheme:
    """Material Design inspired color scheme"""

    PRIMARY = "#2C3E50"
    PRIMARY_DARK = "#1A252F"
    PRIMARY_LIGHT = "#34495E"
    ACCENT = "#3498DB"
    ACCENT_HOVER = "#2980B9"
    SUCCESS = "#27AE60"
    SUCCESS_HOVER = "#229954"
    WARNING = "#F39C12"
    DANGER = "#E74C3C"
    BG_DARK = "#1E1E1E"
    BG_MEDIUM = "#2D2D30"
    BG_LIGHT = "#3E3E42"
    BG_CARD = "#FFFFFF"
    BG_APP = "#F0F3F4"
    TEXT_PRIMARY = "#2C3E50"
    TEXT_SECONDARY = "#7F8C8D"
    TEXT_LIGHT = "#FFFFFF"
    TEXT_MUTED = "#95A5A6"
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 16, "bold")
    FONT_HEADING = (FONT_FAMILY, 12, "bold")
    FONT_BODY = (FONT_FAMILY, 10)
    FONT_SMALL = (FONT_FAMILY, 9)


@dataclass
class MapLayer:
    """Represents a renderable layer of the map"""

    name: str
    image: Image.Image | None = None
    visible: bool = True
    opacity: float = 1.0
    z_order: int = 0


class OptimizedMapCanvas(Canvas):
    """
    Optimized map canvas that pre-renders layers as images for smooth performance.
    Instead of drawing thousands of individual canvas items, we render to PIL images
    and display them as a single canvas image.
    """

    def __init__(self, parent, width=800, height=600):
        # Create container with scrollbars
        self.container = tk.Frame(parent, bg=ModernTheme.BG_DARK)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Create scrollbars
        self.v_scrollbar = tk.Scrollbar(self.container, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scrollbar = tk.Scrollbar(self.container, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create canvas
        super().__init__(
            self.container,
            width=width,
            height=height,
            bg=ModernTheme.BG_DARK,
            highlightthickness=0,
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set,
        )

        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        self.h_scrollbar.config(command=self.xview)
        self.v_scrollbar.config(command=self.yview)

        # Map data
        self.map_data = None
        self.map_width = 250
        self.map_height = 250
        self.tile_size = 4  # Pixels per tile - increased for better visibility

        # Rendering layers
        self.layers = {
            "background": MapLayer("background", z_order=0),
            "terrain": MapLayer("terrain", z_order=1),
            "bridges": MapLayer("bridges", z_order=2),
            "buildings": MapLayer("buildings", z_order=3),
            "colonists": MapLayer("colonists", z_order=4),
            "overlay": MapLayer("overlay", z_order=5),
        }

        # Composite image
        self.composite_image = None
        self.photo_image = None
        self.canvas_image_id = None

        # Interaction state
        self.selection = None
        self.drag_start = None
        self.selection_callback = None
        self.zoom_level = 1.0
        self.view_offset = (0, 0)

        # Performance settings
        self.use_threading = True
        self.render_quality = "high"  # 'low', 'medium', 'high'
        self.cache_enabled = True

        # Bind events
        self._bind_events()

        # Initial render
        self._render_empty_state()

    def _bind_events(self):
        """Bind interaction events"""
        # Selection
        self.bind("<Button-1>", self.start_selection)
        self.bind("<B1-Motion>", self.update_selection)
        self.bind("<ButtonRelease-1>", self.finish_selection)

        # Zoom with mouse wheel
        self.bind("<Control-MouseWheel>", self.on_zoom)  # Windows
        self.bind("<Control-Button-4>", self.on_zoom)  # Linux up
        self.bind("<Control-Button-5>", self.on_zoom)  # Linux down

        # Pan with middle mouse or shift+drag
        self.bind("<Button-2>", self.start_pan)
        self.bind("<B2-Motion>", self.do_pan)
        self.bind("<ButtonRelease-2>", self.end_pan)
        self.bind("<Shift-Button-1>", self.start_pan)
        self.bind("<Shift-B1-Motion>", self.do_pan)
        self.bind("<Shift-ButtonRelease-1>", self.end_pan)

    def _render_empty_state(self):
        """Render initial empty state"""
        img = Image.new("RGB", (800, 600), color="#1E1E1E")
        draw = ImageDraw.Draw(img)

        # Draw help text
        try:
            from PIL import ImageFont

            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, ImportError):
            font = None
            small_font = None

        draw.text(
            (400, 280),
            "Load a save file to see the map",
            fill="#95A5A6",
            anchor="mm",
            font=font,
        )
        draw.text(
            (400, 320),
            "Click and drag to select base location",
            fill="#95A5A6",
            anchor="mm",
            font=small_font,
        )
        draw.text(
            (400, 350),
            "Ctrl+Scroll to zoom, Shift+Drag to pan",
            fill="#95A5A6",
            anchor="mm",
            font=small_font,
        )

        self._display_image(img)

    def load_map(self, game_state, foundation_grid=None):
        """Load and render map from game state"""
        self.map_data = game_state
        self.foundation_grid = foundation_grid

        if not game_state or not game_state.maps:
            self._render_empty_state()
            return

        first_map = game_state.maps[0]
        self.current_save = first_map

        # Get map dimensions
        self.map_width = getattr(first_map, "width", 250)
        self.map_height = getattr(first_map, "height", 250)

        # Start rendering in thread if enabled
        if self.use_threading:
            thread = threading.Thread(
                target=self._render_map_threaded, args=(first_map, foundation_grid)
            )
            thread.daemon = True
            thread.start()
        else:
            self._render_map(first_map, foundation_grid)

    def _render_map_threaded(self, map_data, foundation_grid):
        """Render map in background thread"""
        self._render_map(map_data, foundation_grid)
        # Schedule display update on main thread
        self.after(0, self._update_display)

    def _render_map(self, map_data, foundation_grid):
        """Render all map layers to images"""
        canvas_width = self.map_width * self.tile_size
        canvas_height = self.map_height * self.tile_size

        # Render background layer
        self._render_background_layer(canvas_width, canvas_height)

        # Render bridges layer
        if foundation_grid is not None:
            self._render_bridges_layer(foundation_grid, canvas_width, canvas_height)

        # Render buildings layer
        self._render_buildings_layer(map_data, canvas_width, canvas_height)

        # Render colonists layer
        self._render_colonists_layer(map_data, canvas_width, canvas_height)

        # Composite all layers
        self._composite_layers()

    def _render_background_layer(self, width, height):
        """Render grid background"""
        img = Image.new("RGBA", (width, height), (30, 30, 30, 255))
        draw = ImageDraw.Draw(img)

        # Draw major grid lines
        grid_spacing = self.tile_size * 10
        for i in range(0, width + 1, grid_spacing):
            draw.line([(i, 0), (i, height)], fill=(45, 45, 48, 255), width=1)
        for i in range(0, height + 1, grid_spacing):
            draw.line([(0, i), (width, i)], fill=(45, 45, 48, 255), width=1)

        self.layers["background"].image = img

    def _render_bridges_layer(self, foundation_grid, width, height):
        """Render bridges as a layer"""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Find bridge positions
        light_positions = np.argwhere(foundation_grid == 0x1A47)
        heavy_positions = np.argwhere(foundation_grid == 0x8C7D)

        # Draw heavy bridges
        for y, x in heavy_positions:
            x1 = x * self.tile_size
            y1 = (self.map_height - y - 1) * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            draw.rectangle(
                [x1, y1, x2, y2], fill=(80, 80, 80, 255), outline=(48, 48, 48, 255)
            )

        # Draw light bridges
        for y, x in light_positions:
            x1 = x * self.tile_size
            y1 = (self.map_height - y - 1) * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            draw.rectangle(
                [x1, y1, x2, y2], fill=(139, 69, 19, 255), outline=(107, 52, 16, 255)
            )

        self.layers["bridges"].image = img
        print(
            f"Rendered {len(heavy_positions)} heavy bridges and {len(light_positions)} light bridges"
        )

    def _render_buildings_layer(self, map_data, width, height):
        """Render buildings as separate sub-layers for better control"""
        # Create separate layers for different building types (NO UTILITIES!)
        self.layers["walls"] = MapLayer("walls", z_order=3, visible=True)
        self.layers["doors"] = MapLayer("doors", z_order=4, visible=True)
        self.layers["furniture"] = MapLayer("furniture", z_order=5, visible=True)
        self.layers["storage"] = MapLayer("storage", z_order=6, visible=True)
        self.layers["defense"] = MapLayer("defense", z_order=7, visible=True)

        # Create images for each layer
        layer_images = {
            "walls": Image.new("RGBA", (width, height), (0, 0, 0, 0)),
            "doors": Image.new("RGBA", (width, height), (0, 0, 0, 0)),
            "furniture": Image.new("RGBA", (width, height), (0, 0, 0, 0)),
            "storage": Image.new("RGBA", (width, height), (0, 0, 0, 0)),
            "defense": Image.new("RGBA", (width, height), (0, 0, 0, 0)),
        }

        draws = {name: ImageDraw.Draw(img) for name, img in layer_images.items()}

        # Group buildings by type for batch rendering
        building_groups = {
            "walls": [],
            "doors": [],
            "furniture": [],
            "storage": [],
            "defense": [],
            "other": [],
        }

        # Categorize buildings
        for building in map_data.buildings:
            # SKIP UTILITIES ENTIRELY - don't even load them
            if any(
                util in building.def_name
                for util in [
                    "PowerConduit",
                    "Conduit",
                    "Pipe",
                    "Vent",
                    "Cooler",
                    "Heater",
                    "AirConditioning",
                    "Plumbing",
                    "WaterPipe",
                    "AirPipe",
                ]
            ):
                continue  # Skip completely

            x = building.position.x * self.tile_size
            y = (self.map_height - building.position.y - 1) * self.tile_size

            if "Wall" in building.def_name:
                building_groups["walls"].append((x, y, building))
            elif "Door" in building.def_name:
                building_groups["doors"].append((x, y, building))
            elif any(
                furn in building.def_name
                for furn in ["Bed", "Table", "Chair", "Dresser"]
            ):
                building_groups["furniture"].append((x, y, building))
            elif any(
                pwr in building.def_name
                for pwr in ["Solar", "Battery", "Generator", "Turbine"]
            ):
                # Power generation is important - keep it visible in furniture layer
                building_groups["furniture"].append((x, y, building))
            elif "Storage" in building.def_name or "Shelf" in building.def_name:
                building_groups["storage"].append((x, y, building))
            elif "Turret" in building.def_name or "Trap" in building.def_name:
                building_groups["defense"].append((x, y, building))
            else:
                # Determine best category for other items
                if any(
                    x in building.def_name.lower() for x in ["lamp", "light", "torch"]
                ):
                    building_groups["furniture"].append((x, y, building))
                else:
                    building_groups["walls"].append(
                        (x, y, building)
                    )  # Default to walls layer

        # Render each group with appropriate colors
        colors = {
            "walls": (128, 128, 128, 255),
            "doors": (160, 82, 45, 255),
            "furniture": (65, 105, 225, 255),
            "storage": (255, 215, 0, 255),
            "defense": (255, 0, 0, 255),
            "other": (150, 165, 166, 255),
        }

        sizes = {
            "walls": 2,
            "doors": 3,
            "furniture": 4,
            "storage": 3,
            "defense": 4,
            "other": 2,
        }

        # Draw buildings by group into their respective layers
        for group_name, buildings in building_groups.items():
            if group_name == "other":
                continue  # Skip 'other' since we redistributed them

            color = colors[group_name]
            size = sizes[group_name]
            draw = draws[group_name]

            for x, y, building in buildings:
                # For performance, use simple rectangles
                draw.rectangle(
                    [x - size // 2, y - size // 2, x + size // 2, y + size // 2],
                    fill=color,
                )

        # Assign images to layers
        for layer_name, img in layer_images.items():
            self.layers[layer_name].image = img

        # Log statistics (utilities are completely skipped now)
        total_skipped = len(
            [
                b
                for b in map_data.buildings
                if any(
                    u in b.def_name
                    for u in [
                        "PowerConduit",
                        "Conduit",
                        "Pipe",
                        "Vent",
                        "Cooler",
                        "Heater",
                        "AirConditioning",
                        "Plumbing",
                        "WaterPipe",
                        "AirPipe",
                    ]
                )
            ]
        )

        print("Rendered buildings by layer:")
        print(f"  Walls: {len(building_groups['walls'])}")
        print(f"  Doors: {len(building_groups['doors'])}")
        print(f"  Furniture: {len(building_groups['furniture'])}")
        print(f"  Storage: {len(building_groups['storage'])}")
        print(f"  Defense: {len(building_groups['defense'])}")
        print(
            f"  Skipped utilities: {total_skipped} (conduits, pipes, vents - not loaded)"
        )

    def _render_colonists_layer(self, map_data, width, height):
        """Render colonists as a layer"""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        colonists = (
            map_data.get_colonists() if hasattr(map_data, "get_colonists") else []
        )

        for colonist in colonists:
            if hasattr(colonist, "position"):
                x = colonist.position.x * self.tile_size
                y = (self.map_height - colonist.position.y - 1) * self.tile_size
                # Draw colonist as a green circle
                draw.ellipse(
                    [x - 3, y - 3, x + 3, y + 3],
                    fill=(0, 255, 0, 255),
                    outline=(255, 255, 255, 255),
                )

        self.layers["colonists"].image = img
        print(f"Rendered {len(colonists)} colonists")

    def _composite_layers(self):
        """Composite all visible layers into final image"""
        width = self.map_width * self.tile_size
        height = self.map_height * self.tile_size

        # Create base image
        composite = Image.new("RGBA", (width, height), (30, 30, 30, 255))

        # Sort layers by z_order
        sorted_layers = sorted(self.layers.values(), key=lambda layer: layer.z_order)

        # Composite each visible layer
        for layer in sorted_layers:
            if layer.visible and layer.image:
                if layer.opacity < 1.0:
                    # Apply opacity
                    overlay = layer.image.copy()
                    overlay.putalpha(int(255 * layer.opacity))
                    composite.alpha_composite(overlay)
                else:
                    composite.alpha_composite(layer.image)

        self.composite_image = composite

    def _update_display(self):
        """Update the canvas display with the composite image"""
        if self.composite_image:
            # Apply zoom if needed
            if self.zoom_level != 1.0:
                new_size = (
                    int(self.composite_image.width * self.zoom_level),
                    int(self.composite_image.height * self.zoom_level),
                )
                display_image = self.composite_image.resize(
                    new_size, Image.NEAREST if self.zoom_level > 1 else Image.LANCZOS
                )
            else:
                display_image = self.composite_image

            self._display_image(display_image)

            # Update scroll region
            self.config(scrollregion=(0, 0, display_image.width, display_image.height))

    def _display_image(self, image):
        """Display PIL image on canvas"""
        self.photo_image = ImageTk.PhotoImage(image)

        # Remove old image if exists
        if self.canvas_image_id:
            self.delete(self.canvas_image_id)

        # Create new image
        self.canvas_image_id = self.create_image(
            0, 0, anchor=tk.NW, image=self.photo_image
        )

        # Ensure selection stays on top
        if self.selection:
            self.tag_raise(self.selection)

    def set_layer_visibility(self, layer_name: str, visible: bool):
        """Toggle layer visibility"""
        if layer_name in self.layers:
            self.layers[layer_name].visible = visible
            self._composite_layers()
            self._update_display()

    def set_zoom(self, zoom_level: float):
        """Set zoom level"""
        self.zoom_level = max(0.25, min(4.0, zoom_level))
        self._update_display()

    def on_zoom(self, event):
        """Handle zoom with mouse wheel"""
        # Get mouse position in canvas coordinates
        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        else:
            zoom_factor = 0.9

        # Apply zoom
        self.set_zoom(self.zoom_level * zoom_factor)

        # Adjust view to keep mouse position centered
        self.xview_moveto(
            (canvas_x * zoom_factor - event.x)
            / (self.composite_image.width * self.zoom_level)
        )
        self.yview_moveto(
            (canvas_y * zoom_factor - event.y)
            / (self.composite_image.height * self.zoom_level)
        )

    def start_pan(self, event):
        """Start panning"""
        self.scan_mark(event.x, event.y)

    def do_pan(self, event):
        """Pan the view"""
        self.scan_dragto(event.x, event.y, gain=1)

    def end_pan(self, event):
        """End panning"""
        pass

    def start_selection(self, event):
        """Start area selection"""
        if event.state & 0x1:  # Shift key
            return

        canvas_x = self.canvasx(event.x)
        canvas_y = self.canvasy(event.y)
        self.drag_start = (canvas_x, canvas_y)

        self.delete("selection")
        self.selection = self.create_rectangle(
            canvas_x,
            canvas_y,
            canvas_x,
            canvas_y,
            outline="#3498DB",
            width=2,
            dash=(5, 5),
            tags="selection",
        )

    def update_selection(self, event):
        """Update selection rectangle"""
        if self.selection and self.drag_start:
            canvas_x = self.canvasx(event.x)
            canvas_y = self.canvasy(event.y)
            x1, y1 = self.drag_start
            self.coords(self.selection, x1, y1, canvas_x, canvas_y)

    def finish_selection(self, event):
        """Finish selection"""
        if self.drag_start and self.selection_callback:
            canvas_x = self.canvasx(event.x)
            canvas_y = self.canvasy(event.y)
            x1, y1 = self.drag_start

            # Convert to map coordinates
            map_x1 = int(x1 / (self.tile_size * self.zoom_level))
            map_y1 = int(y1 / (self.tile_size * self.zoom_level))
            map_x2 = int(canvas_x / (self.tile_size * self.zoom_level))
            map_y2 = int(canvas_y / (self.tile_size * self.zoom_level))

            # Ensure correct ordering
            map_x1, map_x2 = min(map_x1, map_x2), max(map_x1, map_x2)
            map_y1, map_y2 = min(map_y1, map_y2), max(map_y1, map_y2)

            # Callback with selection
            if self.selection_callback:
                self.selection_callback(
                    map_x1, map_y1, map_x2 - map_x1, map_y2 - map_y1
                )

        self.drag_start = None

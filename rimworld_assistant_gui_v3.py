#!/usr/bin/env python3
"""
RimWorld Base Assistant - Modern Professional GUI
Enhanced interface with Material Design inspired aesthetics and full feature set.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Canvas
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import json
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.save_parser import RimWorldSaveParser
from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator, PrefabUsageMode
from src.generators.requirements_driven_generator import RequirementsDrivenGenerator
from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.ai.claude_base_designer import ClaudeBaseDesigner, BaseDesignRequest
from src.visualization.base_visualizer import BaseVisualizer
from src.visualization.layered_visualizer import LayeredVisualizer
from src.utils.progress import StepProgress
from src.utils.symbols import SUCCESS, FAILURE, WARNING


class ModernTheme:
    """Material Design inspired color scheme"""
    # Primary colors
    PRIMARY = "#2C3E50"        # Dark blue-gray
    PRIMARY_DARK = "#1A252F"   # Darker variant
    PRIMARY_LIGHT = "#34495E"  # Lighter variant
    
    # Accent colors
    ACCENT = "#3498DB"          # Bright blue
    ACCENT_HOVER = "#2980B9"    # Darker blue on hover
    SUCCESS = "#27AE60"         # Green
    SUCCESS_HOVER = "#229954"   # Darker green
    WARNING = "#F39C12"         # Orange
    DANGER = "#E74C3C"          # Red
    
    # Background colors
    BG_DARK = "#1E1E1E"         # Very dark
    BG_MEDIUM = "#2D2D30"       # Medium dark
    BG_LIGHT = "#3E3E42"        # Light dark
    BG_CARD = "#FFFFFF"         # White cards
    BG_APP = "#F0F3F4"          # Light gray app background
    
    # Text colors
    TEXT_PRIMARY = "#2C3E50"    # Dark text on light
    TEXT_SECONDARY = "#7F8C8D"  # Gray text
    TEXT_LIGHT = "#FFFFFF"      # White text on dark
    TEXT_MUTED = "#95A5A6"      # Muted gray
    
    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 16, "bold")
    FONT_HEADING = (FONT_FAMILY, 12, "bold")
    FONT_BODY = (FONT_FAMILY, 10)
    FONT_SMALL = (FONT_FAMILY, 9)


class StyledButton(tk.Button):
    """Modern styled button with hover effects"""
    def __init__(self, parent, text="", command=None, style="primary", width=None, **kwargs):
        styles = {
            "primary": (ModernTheme.ACCENT, ModernTheme.ACCENT_HOVER, ModernTheme.TEXT_LIGHT),
            "success": (ModernTheme.SUCCESS, ModernTheme.SUCCESS_HOVER, ModernTheme.TEXT_LIGHT),
            "danger": (ModernTheme.DANGER, "#C0392B", ModernTheme.TEXT_LIGHT),
            "secondary": (ModernTheme.BG_LIGHT, ModernTheme.PRIMARY, ModernTheme.TEXT_LIGHT),
            "outline": (ModernTheme.BG_CARD, ModernTheme.BG_APP, ModernTheme.ACCENT)
        }
        
        bg, hover, fg = styles.get(style, styles["primary"])
        
        config = {
            "text": text,
            "command": command,
            "bg": bg,
            "fg": fg,
            "font": ModernTheme.FONT_BODY,
            "relief": tk.FLAT,
            "padx": 20,
            "pady": 10,
            "cursor": "hand2",
            "bd": 0,
            "highlightthickness": 0
        }
        
        if width:
            config["width"] = width
            
        config.update(kwargs)
        super().__init__(parent, **config)
        
        self.default_bg = bg
        self.hover_bg = hover
        
        self.bind("<Enter>", lambda e: self.config(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self.default_bg))


class ModeCard(tk.Frame):
    """Beautiful card for mode selection with animations"""
    def __init__(self, parent, title, description, icon, value, variable, command=None, app=None):
        super().__init__(parent, bg=ModernTheme.BG_CARD, relief=tk.FLAT, bd=0)
        
        self.value = value
        self.variable = variable
        self.command = command
        self.selected = False
        self.app = app
        
        # Card styling
        self.config(highlightbackground=ModernTheme.BG_APP, highlightthickness=2)
        
        # Content with padding
        content = tk.Frame(self, bg=ModernTheme.BG_CARD)
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # Header with icon
        header = tk.Frame(content, bg=ModernTheme.BG_CARD)
        header.pack(fill=tk.X)
        
        # Icon
        icon_label = tk.Label(header, text=icon, font=(ModernTheme.FONT_FAMILY, 24), 
                              bg=ModernTheme.BG_CARD, fg=ModernTheme.ACCENT)
        icon_label.pack(side=tk.LEFT, padx=(0, 12))
        
        # Title
        title_label = tk.Label(header, text=title, font=ModernTheme.FONT_HEADING,
                              bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY)
        title_label.pack(side=tk.LEFT)
        
        # Description
        desc_label = tk.Label(content, text=description, font=ModernTheme.FONT_SMALL,
                             bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_SECONDARY,
                             wraplength=280, justify=tk.LEFT, anchor=tk.W)
        desc_label.pack(fill=tk.X, pady=(8, 0))
        
        # Bind click events
        self.bind_all_children(self)
        
    def bind_all_children(self, widget):
        """Bind click events to all children"""
        widget.bind("<Button-1>", self.on_click)
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
        for child in widget.winfo_children():
            self.bind_all_children(child)
    
    def on_click(self, event):
        """Handle click event"""
        self.variable.set(self.value)
        if self.command:
            self.command()
        if self.app:
            self.app.update_selection()
    
    def on_enter(self, event):
        """Mouse enter effect"""
        if not self.selected:
            self.config(highlightbackground=ModernTheme.ACCENT, highlightthickness=2)
    
    def on_leave(self, event):
        """Mouse leave effect"""
        if not self.selected:
            self.config(highlightbackground=ModernTheme.BG_APP, highlightthickness=2)
    
    def set_selected(self, selected):
        """Update selection state"""
        self.selected = selected
        if selected:
            self.config(highlightbackground=ModernTheme.SUCCESS, highlightthickness=3)
        else:
            self.config(highlightbackground=ModernTheme.BG_APP, highlightthickness=2)


class InteractiveMapCanvas(Canvas):
    """Enhanced map canvas with better visuals and interaction"""
    def __init__(self, parent, width=600, height=600):
        super().__init__(parent, width=width, height=height, 
                        bg=ModernTheme.BG_DARK, highlightthickness=0)
        
        self.map_data = None
        self.selection = None
        self.drag_start = None
        self.selection_callback = None
        self.grid_size = 10  # Size of each grid cell in pixels
        
        # Bind events
        self.bind("<Button-1>", self.start_selection)
        self.bind("<B1-Motion>", self.update_selection)
        self.bind("<ButtonRelease-1>", self.finish_selection)
        self.bind("<Motion>", self.on_mouse_move)
        
        # Initial drawing
        self.draw_background()
        self.draw_help_text()
        
    def draw_background(self):
        """Draw grid background"""
        # Clear canvas
        self.delete("all")
        
        # Draw grid
        for i in range(0, 600, self.grid_size * 5):
            # Major grid lines
            self.create_line(i, 0, i, 600, fill=ModernTheme.BG_MEDIUM, width=1)
            self.create_line(0, i, 600, i, fill=ModernTheme.BG_MEDIUM, width=1)
        
        for i in range(0, 600, self.grid_size):
            # Minor grid lines
            if i % (self.grid_size * 5) != 0:
                self.create_line(i, 0, i, 600, fill="#252525", width=1, dash=(2, 4))
                self.create_line(0, i, 600, i, fill="#252525", width=1, dash=(2, 4))
    
    def draw_help_text(self):
        """Draw help text when no map is loaded"""
        if not self.map_data:
            self.create_text(300, 280, text="Load a save file to see the map",
                           font=ModernTheme.FONT_HEADING, fill=ModernTheme.TEXT_MUTED)
            self.create_text(300, 320, text="Click and drag to select base location",
                           font=ModernTheme.FONT_BODY, fill=ModernTheme.TEXT_MUTED)
    
    def load_map(self, game_state):
        """Load and visualize map from game state"""
        self.map_data = game_state
        self.draw_background()
        
        if game_state and game_state.maps:
            first_map = game_state.maps[0]
            
            # Calculate scale
            map_width = first_map.width if hasattr(first_map, 'width') else 250
            map_height = first_map.height if hasattr(first_map, 'height') else 250
            scale_x = 600 / map_width
            scale_y = 600 / map_height
            
            # Draw terrain (simplified)
            terrain_colors = {
                "Mountain": "#4A4A4A",
                "Water": "#2E86AB",
                "Marsh": "#4A7C59",
                "Sand": "#F4E4C1",
                "Soil": "#8B7355",
                "SoilRich": "#6B5B45"
            }
            
            # Count building types for debugging
            bridge_count = 0
            frame_count = 0
            
            # Draw buildings
            for building in first_map.buildings:
                if "Bridge" in building.def_name:
                    bridge_count += 1
                if building.is_frame:
                    frame_count += 1
                x = int(building.position.x * scale_x)
                # Fix Y-axis flip - RimWorld Y increases downward, Canvas Y increases downward too
                # But the map appears flipped, so we need to invert Y
                y = int((map_height - building.position.y - 1) * scale_y)
                
                # Determine color based on building type
                color = ModernTheme.TEXT_MUTED
                size = 2
                
                if "Wall" in building.def_name:
                    color = "#808080"
                    size = 2
                elif "Door" in building.def_name:
                    color = "#A0522D"
                    size = 3
                elif "Bed" in building.def_name:
                    color = "#4169E1"
                    size = 4
                elif "Table" in building.def_name:
                    color = "#8B4513"
                    size = 3
                elif "Storage" in building.def_name:
                    color = "#FFD700"
                    size = 3
                elif "Power" in building.def_name or "Battery" in building.def_name:
                    color = "#32CD32"
                    size = 3
                elif "Turret" in building.def_name:
                    color = "#FF0000"
                    size = 4
                elif building.def_name == "HeavyBridge":
                    # Completed heavy bridge - bright green with thick border
                    color = "#00FF00"  # Bright green for completed
                    size = 8  # Large size
                    # Draw with thick outline for visibility
                    self.create_rectangle(x-size//2-1, y-size//2-1, x+size//2+1, y+size//2+1,
                                         fill=color, outline="#FFFFFF", width=2, tags="bridge_complete")
                    continue  # Skip the default rectangle drawing
                elif "Frame_HeavyBridge" in building.def_name:
                    # Bridge blueprint/frame - cyan with dashed outline
                    color = "#00FFFF"  # Cyan for frames
                    size = 6
                    self.create_rectangle(x-size//2, y-size//2, x+size//2, y+size//2,
                                         fill=color, outline="#FFFFFF", width=1, dash=(3,3), tags="bridge_frame")
                    continue  # Skip the default rectangle drawing
                elif "Bridge" in building.def_name:
                    # Other bridge types
                    color = "#40E0D0"  # Turquoise
                    size = 6
                elif building.is_frame:
                    color = "#FFFF00"  # Yellow for other frames
                    size = 4
                
                # Draw building as small rectangle (for non-bridge items)
                self.create_rectangle(x-size//2, y-size//2, x+size//2, y+size//2,
                                     fill=color, outline="", tags="building")
            
            # Log what we found
            print(f"Map loaded: Found {bridge_count} bridges and {frame_count} frames out of {len(first_map.buildings)} total buildings")
            print("Note: Completed heavy bridges are stored as terrain tiles which are compressed - only showing bridge frames/blueprints")
            
            # Draw colonists
            if hasattr(first_map, 'colonists'):
                for colonist in first_map.colonists:
                    if hasattr(colonist, 'position'):
                        x = int(colonist.position.x * scale_x)
                        # Apply same Y-flip as buildings
                        y = int((map_height - colonist.position.y - 1) * scale_y)
                        self.create_oval(x-3, y-3, x+3, y+3,
                                       fill="#00FF00", outline="#FFFFFF", width=1, tags="colonist")
    
    def start_selection(self, event):
        """Start area selection"""
        self.drag_start = (event.x, event.y)
        
        # Remove old selection
        self.delete("selection")
        
        # Create new selection rectangle
        self.selection = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=ModernTheme.ACCENT, width=2, dash=(5, 5),
            tags="selection"
        )
    
    def update_selection(self, event):
        """Update selection rectangle while dragging"""
        if self.selection and self.drag_start:
            x1, y1 = self.drag_start
            self.coords(self.selection, x1, y1, event.x, event.y)
    
    def finish_selection(self, event):
        """Finish selection and calculate area"""
        if self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            
            # Ensure correct ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Convert to map coordinates
            map_coords = (
                int(x1 * 250 / 600),
                int(y1 * 250 / 600),
                int(x2 * 250 / 600),
                int(y2 * 250 / 600)
            )
            
            # Calculate size
            width = map_coords[2] - map_coords[0]
            height = map_coords[3] - map_coords[1]
            
            # Update selection display
            if self.selection:
                self.itemconfig(self.selection, outline=ModernTheme.SUCCESS, width=3)
            
            # Callback with selection info
            if self.selection_callback:
                self.selection_callback(map_coords, width, height)
            
            self.drag_start = None
    
    def on_mouse_move(self, event):
        """Show coordinates on mouse move"""
        map_x = int(event.x * 250 / 600)
        map_y = int(event.y * 250 / 600)
        
        # Update cursor coordinates (if we have a label for it)
        if hasattr(self.master, 'coord_label'):
            self.master.coord_label.config(text=f"Map coordinates: ({map_x}, {map_y})")


class RimWorldAssistantGUI:
    """Main GUI application with modern design"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RimWorld Base Assistant - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg=ModernTheme.BG_APP)
        
        # Set window icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Variables
        self.generation_mode = tk.StringVar(value="smart")
        self.base_width = tk.IntVar(value=60)
        self.base_height = tk.IntVar(value=60)
        self.game_state = None
        self.save_path = None
        self.last_grid = None
        self.selected_area = None
        self.current_view = "map"  # "map" or "generated"
        
        # Visualizer with layer support
        self.visualizer = LayeredVisualizer(scale=10)
        self.layer_vars = {}
        
        # Check for AlphaPrefabs
        self.alpha_prefabs_path = self.find_alpha_prefabs()
        
        # Setup UI
        self.setup_ui()
        
    def find_alpha_prefabs(self):
        """Find AlphaPrefabs mod directory"""
        possible_paths = [
            Path("data/AlphaPrefabs"),
            Path("../AlphaPrefabs"),
            Path("../../AlphaPrefabs"),
            Path.home() / "AlphaPrefabs"
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "PrefabSets").exists():
                return path
        return None
    
    def setup_ui(self):
        """Setup the user interface"""
        # Top toolbar
        self.setup_toolbar()
        
        # Main content area
        content = tk.Frame(self.root, bg=ModernTheme.BG_APP)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Three column layout
        self.setup_left_panel(content)
        self.setup_center_panel(content)
        self.setup_right_panel(content)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_toolbar(self):
        """Setup top toolbar"""
        toolbar = tk.Frame(self.root, bg=ModernTheme.PRIMARY, height=60)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)
        
        # Title
        title = tk.Label(toolbar, text="🏰 RimWorld Base Assistant", 
                        font=(ModernTheme.FONT_FAMILY, 18, "bold"),
                        bg=ModernTheme.PRIMARY, fg=ModernTheme.TEXT_LIGHT)
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Toolbar buttons
        btn_frame = tk.Frame(toolbar, bg=ModernTheme.PRIMARY)
        btn_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        StyledButton(btn_frame, "📁 Load Save", self.load_save, "outline", width=12).pack(side=tk.LEFT, padx=5)
        StyledButton(btn_frame, "💾 Export", self.export_base, "outline", width=10).pack(side=tk.LEFT, padx=5)
        StyledButton(btn_frame, "ℹ️ Help", self.show_help, "outline", width=8).pack(side=tk.LEFT, padx=5)
    
    def setup_left_panel(self, parent):
        """Setup left panel with generation modes"""
        left_frame = tk.Frame(parent, bg=ModernTheme.BG_APP, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Section header
        header = tk.Label(left_frame, text="Generation Modes", 
                         font=ModernTheme.FONT_TITLE,
                         bg=ModernTheme.BG_APP, fg=ModernTheme.TEXT_PRIMARY)
        header.pack(pady=(0, 15))
        
        # Mode cards container
        self.cards_container = tk.Frame(left_frame, bg=ModernTheme.BG_APP)
        self.cards_container.pack(fill=tk.BOTH, expand=True)
        
        # Mode definitions
        modes = [
            {
                "title": "Smart Generate",
                "desc": "Best option! Uses NLP to understand your requirements, then intelligently selects and places real RimWorld prefabs using best practices.",
                "icon": "🎯",
                "value": "smart"
            },
            {
                "title": "Natural Language",
                "desc": "Describe your base in plain English. Uses advanced algorithms for procedural generation.",
                "icon": "💬",
                "value": "nlp"
            },
            {
                "title": "Prefab Anchors",
                "desc": "Places complete room designs from AlphaPrefabs, then fills remaining space.",
                "icon": "🏗️",
                "value": "prefab"
            },
            {
                "title": "Enhanced Hybrid",
                "desc": "Combines multiple generation techniques for maximum variety and realism.",
                "icon": "🔀",
                "value": "hybrid"
            },
            {
                "title": "AI Designer",
                "desc": "Uses Claude AI to create detailed base specifications following best practices.",
                "icon": "🤖",
                "value": "ai"
            }
        ]
        
        # Create mode cards
        self.mode_cards = []
        for mode in modes:
            card = ModeCard(self.cards_container, 
                          mode["title"], mode["desc"], mode["icon"],
                          mode["value"], self.generation_mode,
                          self.on_mode_change, app=self)
            card.pack(fill=tk.X, pady=5)
            self.mode_cards.append(card)
        
        # Set initial selection
        self.update_selection()
    
    def setup_center_panel(self, parent):
        """Setup center panel with input and controls"""
        center_frame = tk.Frame(parent, bg=ModernTheme.BG_CARD, relief=tk.FLAT)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Inner padding
        inner = tk.Frame(center_frame, bg=ModernTheme.BG_CARD)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Requirements input section
        tk.Label(inner, text="Base Requirements", font=ModernTheme.FONT_TITLE,
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY).pack(anchor=tk.W)
        
        tk.Label(inner, text="Describe your ideal base in detail:",
                font=ModernTheme.FONT_BODY,
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_SECONDARY).pack(anchor=tk.W, pady=(5, 10))
        
        # Large text input area
        text_frame = tk.Frame(inner, bg=ModernTheme.BG_CARD, relief=tk.SOLID, bd=1)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_input = scrolledtext.ScrolledText(
            text_frame,
            height=12,
            font=(ModernTheme.FONT_FAMILY, 11),
            wrap=tk.WORD,
            bg="#FAFAFA",
            fg=ModernTheme.TEXT_PRIMARY,
            insertbackground=ModernTheme.ACCENT,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # Default text with comprehensive example
        default_text = """Create a defensive mountain base for 12 colonists with:

ESSENTIAL ROOMS:
• Individual bedrooms for each colonist (using Realistic Rooms mod dimensions)
• Central dining hall with tables and chairs
• Kitchen with freezer attached
• Medical bay near entrance for quick rescue
• Workshop cluster (crafting, tailoring, smithy) with shared tool storage
• Research lab with hi-tech benches
• Recreation room in central location

DEFENSE:
• Killbox entrance with overlapping turret fields
• Defensive positions with sandbags
• Multiple fallback points
• Separate prison area

STORAGE:
• Main warehouse for general items
• Armory for weapons and armor
• Medicine storage near hospital
• Raw materials storage near workshops

POWER & UTILITIES:
• Geothermal or solar power generation
• Battery room with fire suppression
• Climate control for all rooms

Make it efficient with good traffic flow and follow RimWorld best practices."""
        
        self.text_input.insert("1.0", default_text)
        
        # Dimension controls
        dim_frame = tk.Frame(inner, bg=ModernTheme.BG_CARD)
        dim_frame.pack(fill=tk.X, pady=(20, 0))
        
        tk.Label(dim_frame, text="Base Dimensions", font=ModernTheme.FONT_HEADING,
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        
        # Custom size controls
        size_controls = tk.Frame(dim_frame, bg=ModernTheme.BG_CARD)
        size_controls.pack(anchor=tk.W)
        
        tk.Label(size_controls, text="Width:", bg=ModernTheme.BG_CARD).grid(row=0, column=0, sticky=tk.W)
        self.width_spin = tk.Spinbox(size_controls, from_=20, to=200, textvariable=self.base_width,
                                     width=8, font=ModernTheme.FONT_BODY)
        self.width_spin.grid(row=0, column=1, padx=(5, 20))
        
        tk.Label(size_controls, text="Height:", bg=ModernTheme.BG_CARD).grid(row=0, column=2, sticky=tk.W)
        self.height_spin = tk.Spinbox(size_controls, from_=20, to=200, textvariable=self.base_height,
                                      width=8, font=ModernTheme.FONT_BODY)
        self.height_spin.grid(row=0, column=3, padx=5)
        
        tk.Label(size_controls, text="tiles", bg=ModernTheme.BG_CARD, 
                fg=ModernTheme.TEXT_SECONDARY).grid(row=0, column=4, padx=(5, 0))
        
        # Preset buttons
        preset_frame = tk.Frame(dim_frame, bg=ModernTheme.BG_CARD)
        preset_frame.pack(anchor=tk.W, pady=(10, 0))
        
        tk.Label(preset_frame, text="Quick sizes:", bg=ModernTheme.BG_CARD,
                fg=ModernTheme.TEXT_SECONDARY).pack(side=tk.LEFT, padx=(0, 10))
        
        for name, w, h in [("Tiny 30×30", 30, 30), ("Small 50×50", 50, 50), 
                          ("Medium 75×75", 75, 75), ("Large 100×100", 100, 100),
                          ("Huge 150×150", 150, 150)]:
            btn = tk.Button(preset_frame, text=name, 
                          command=lambda width=w, height=h: self.set_dimensions(width, height),
                          font=ModernTheme.FONT_SMALL, relief=tk.FLAT,
                          bg="#E8F4F8", fg=ModernTheme.ACCENT,
                          padx=10, pady=5, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=2)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=ModernTheme.ACCENT, fg=ModernTheme.TEXT_LIGHT))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#E8F4F8", fg=ModernTheme.ACCENT))
        
        # Generate button
        gen_frame = tk.Frame(inner, bg=ModernTheme.BG_CARD)
        gen_frame.pack(fill=tk.X, pady=(30, 0))
        
        self.generate_btn = StyledButton(gen_frame, "🚀 Generate Base", 
                                        self.generate_base, "success")
        self.generate_btn.pack()
        
        # Progress indicator (hidden initially)
        self.progress_frame = tk.Frame(gen_frame, bg=ModernTheme.BG_CARD)
        self.progress_label = tk.Label(self.progress_frame, text="", 
                                      font=ModernTheme.FONT_BODY,
                                      bg=ModernTheme.BG_CARD, fg=ModernTheme.ACCENT)
        self.progress_label.pack()
    
    def setup_right_panel(self, parent):
        """Setup right panel with map and visualization"""
        right_frame = tk.Frame(parent, bg=ModernTheme.BG_CARD, relief=tk.FLAT)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Inner padding
        inner = tk.Frame(right_frame, bg=ModernTheme.BG_CARD)
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(inner, text="Map Visualization", font=ModernTheme.FONT_TITLE,
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY).pack(anchor=tk.W)
        
        # Instructions
        tk.Label(inner, text="Load a save file to see your current base, or generate a new one",
                font=ModernTheme.FONT_BODY,
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_SECONDARY).pack(anchor=tk.W, pady=(5, 15))
        
        # Horizontal split - map and layers side by side
        split_frame = tk.Frame(inner, bg=ModernTheme.BG_CARD)
        split_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Map canvas
        map_frame = tk.Frame(split_frame, bg=ModernTheme.BG_CARD)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Frame(map_frame, bg=ModernTheme.BG_DARK, relief=tk.SUNKEN, bd=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.map_canvas = InteractiveMapCanvas(canvas_frame)
        self.map_canvas.pack(padx=2, pady=2)
        self.map_canvas.selection_callback = self.on_area_selected
        
        # Right side - Layer controls (collapsible)
        layer_frame = tk.Frame(split_frame, bg=ModernTheme.BG_APP, width=250, relief=tk.RIDGE, bd=1)
        layer_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        layer_frame.pack_propagate(False)
        
        # Layer controls
        self.setup_layer_controls(layer_frame)
        
        # Coordinate display
        self.coord_label = tk.Label(inner, text="Map coordinates: (0, 0)",
                                   font=ModernTheme.FONT_SMALL,
                                   bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED)
        self.coord_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Selection info
        self.selection_label = tk.Label(inner, text="No area selected",
                                       font=ModernTheme.FONT_BODY,
                                       bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_SECONDARY)
        self.selection_label.pack(anchor=tk.W, pady=(5, 0))
    
    def setup_layer_controls(self, parent):
        """Setup layer visibility controls"""
        # Title with smaller font for side panel
        tk.Label(parent, text="Layer Controls", font=ModernTheme.FONT_HEADING,
                bg=ModernTheme.BG_APP, fg=ModernTheme.TEXT_PRIMARY).pack(pady=(10, 5))
        
        # Clarification text
        tk.Label(parent, text="(For generated bases)", font=(ModernTheme.FONT_FAMILY, 8),
                bg=ModernTheme.BG_APP, fg=ModernTheme.TEXT_MUTED).pack(pady=(0, 5))
        
        # Layer checkboxes frame - single column for side panel
        controls_frame = tk.Frame(parent, bg=ModernTheme.BG_APP)
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get layer info from visualizer
        layers = self.visualizer.get_layer_info()
        
        # Create checkbox for each layer in single column
        for layer in layers:
            
            # Create variable for this layer
            var = tk.BooleanVar(value=layer["enabled"])
            self.layer_vars[layer["id"]] = var
            
            # Frame for each layer control
            layer_frame = tk.Frame(controls_frame, bg=ModernTheme.BG_APP)
            layer_frame.pack(fill=tk.X, pady=3)
            
            # Color indicator (smaller for side panel)
            color_label = tk.Label(layer_frame, text="■", 
                                 font=(ModernTheme.FONT_FAMILY, 12),
                                 fg=f"#{layer['color'][0]:02x}{layer['color'][1]:02x}{layer['color'][2]:02x}",
                                 bg=ModernTheme.BG_APP)
            color_label.pack(side=tk.LEFT, padx=(0, 5))
            
            # Checkbox (smaller font for side panel)
            cb = tk.Checkbutton(layer_frame, text=layer["name"],
                               variable=var,
                               font=ModernTheme.FONT_SMALL,
                               bg=ModernTheme.BG_APP,
                               fg=ModernTheme.TEXT_PRIMARY,
                               selectcolor=ModernTheme.BG_APP,
                               activebackground=ModernTheme.BG_APP,
                               command=lambda lid=layer["id"]: self.on_layer_toggle(lid))
            cb.pack(side=tk.LEFT)
        
        # Control buttons (smaller for side panel)
        btn_frame = tk.Frame(parent, bg=ModernTheme.BG_APP)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="All", command=self.show_all_layers, 
                 font=ModernTheme.FONT_SMALL, width=6,
                 bg="#5A6268", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="None", command=self.hide_all_layers,
                 font=ModernTheme.FONT_SMALL, width=6,
                 bg="#5A6268", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Apply", command=self.update_layer_preview,
                 font=ModernTheme.FONT_SMALL, width=6,
                 bg=ModernTheme.ACCENT, fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        
        # Status text (removed preview label since we update main canvas)
    
    def on_layer_toggle(self, layer_id):
        """Handle layer visibility toggle"""
        enabled = self.layer_vars[layer_id].get()
        self.visualizer.set_layer_visibility(layer_id, enabled)
        # Auto-update preview when toggling layers (only for generated bases)
        if self.last_grid is not None and self.current_view == "generated":
            self.update_layer_preview()
    
    def show_all_layers(self):
        """Enable all layers"""
        for layer_id, var in self.layer_vars.items():
            var.set(True)
            self.visualizer.set_layer_visibility(layer_id, True)
        self.update_layer_preview()
    
    def hide_all_layers(self):
        """Disable all layers"""
        for layer_id, var in self.layer_vars.items():
            var.set(False)
            self.visualizer.set_layer_visibility(layer_id, False)
        self.update_layer_preview()
    
    def update_layer_preview(self):
        """Update the preview with current layer settings"""
        if self.last_grid is not None and self.current_view == "generated":
            # Only update if we're viewing a generated base
            # Create visualization with current layer settings
            img = self.visualizer.visualize(self.last_grid, 
                                          show_legend=True, flip_y=False)
            
            # Create smaller version for display
            display_size = (600, 600)
            display_img = img.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_img)
            
            # Update canvas
            self.map_canvas.delete("all")
            self.map_canvas.create_image(300, 300, image=photo)
            self.map_canvas.image = photo  # Keep reference
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = tk.Label(self.root, text="Ready - AlphaPrefabs " + 
                                  ("✓ Found" if self.alpha_prefabs_path else "✗ Not found"),
                                  font=ModernTheme.FONT_SMALL,
                                  bg=ModernTheme.PRIMARY_DARK, fg=ModernTheme.TEXT_LIGHT,
                                  anchor=tk.W, padx=15, pady=8)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_selection(self):
        """Update mode card selection states"""
        current_mode = self.generation_mode.get()
        for card in self.mode_cards:
            card.set_selected(card.value == current_mode)
    
    def on_mode_change(self):
        """Handle mode change"""
        self.update_selection()
        mode = self.generation_mode.get()
        self.update_status(f"Selected mode: {mode}")
    
    def on_area_selected(self, coords, width, height):
        """Handle map area selection"""
        self.selected_area = coords
        self.base_width.set(width)
        self.base_height.set(height)
        self.selection_label.config(
            text=f"Selected area: {width}×{height} tiles at position ({coords[0]}, {coords[1]})",
            fg=ModernTheme.SUCCESS
        )
        self.update_status(f"Selected {width}×{height} area on map")
    
    def set_dimensions(self, width, height):
        """Set base dimensions"""
        self.base_width.set(width)
        self.base_height.set(height)
        self.update_status(f"Base size: {width}×{height}")
    
    def load_save(self):
        """Load RimWorld save file"""
        file_path = filedialog.askopenfilename(
            title="Select RimWorld Save File",
            filetypes=[("RimWorld Saves", "*.rws"), ("All Files", "*.*")],
            initialdir="data/saves"
        )
        
        if file_path:
            self.update_status("Loading save file...")
            
            def load():
                try:
                    parser = RimWorldSaveParser()
                    self.game_state = parser.parse(file_path)
                    self.save_path = Path(file_path)
                    
                    # Update UI in main thread
                    self.root.after(0, self.on_save_loaded)
                    
                except Exception as e:
                    self.root.after(0, lambda: self.update_status(f"Error: {str(e)}", "error"))
            
            thread = threading.Thread(target=load)
            thread.daemon = True
            thread.start()
    
    def on_save_loaded(self):
        """Handle save file loaded"""
        if self.game_state and self.game_state.maps:
            first_map = self.game_state.maps[0]
            self.map_canvas.load_map(self.game_state)
            self.current_view = "map"  # Set to map view when loading save
            
            # Count colonists and update default text
            colonist_count = len(first_map.colonists) if hasattr(first_map, 'colonists') else 10
            suggested_count = int(colonist_count * 1.25)  # +25% for growth
            
            # Update the text input with colonist-appropriate default
            self.text_input.delete("1.0", tk.END)
            default_text = f"""Create a defensive mountain base for {suggested_count} colonists (currently have {colonist_count}) with:

ESSENTIAL ROOMS:
• Individual bedrooms for each colonist (using Realistic Rooms mod dimensions)
• Central dining hall with tables and chairs
• Kitchen with freezer attached
• Medical bay near entrance for quick rescue
• Workshop cluster (crafting, tailoring, smithy) with shared tool storage
• Research lab with hi-tech benches
• Recreation room in central location

DEFENSE:
• Killbox entrance with overlapping turret fields
• Defensive positions with sandbags
• Multiple fallback points
• Separate prison area

STORAGE:
• Main warehouse for general items
• Armory for weapons and armor
• Medicine storage near hospital
• Raw materials storage near workshops

POWER & UTILITIES:
• Geothermal or solar power generation
• Battery room with fire suppression
• Climate control for all rooms

Make it efficient with good traffic flow and follow RimWorld best practices."""
            
            self.text_input.insert("1.0", default_text)
            
            self.update_status(f"Loaded: {self.save_path.name} ({colonist_count} colonists, {len(first_map.buildings)} buildings)")
        else:
            self.update_status("Failed to parse save file", "error")
    
    def generate_base(self):
        """Generate base with selected settings"""
        mode = self.generation_mode.get()
        description = self.text_input.get("1.0", tk.END).strip()
        width = self.base_width.get()
        height = self.base_height.get()
        
        if not description:
            messagebox.showwarning("Input Required", 
                                  "Please describe your base requirements")
            return
        
        # Show progress
        self.progress_frame.pack(pady=(10, 0))
        self.progress_label.config(text="🔄 Generating base...")
        self.generate_btn.config(state=tk.DISABLED)
        self.update_status(f"Generating {width}×{height} base...")
        
        def generate():
            try:
                # Initialize progress tracker
                progress = StepProgress(8)
                
                def update_progress(step, message):
                    self.root.after(0, lambda: self.progress_label.config(text=message))
                
                progress.next("Parsing requirements...")
                update_progress(1, "📝 Parsing requirements...")
                
                # Parse requirements
                nlp = BaseGeneratorNLP()
                requirements = nlp.parse_request(description)
                
                grid = None
                
                if mode == "smart":
                    progress.next("Selecting prefabs...")
                    update_progress(2, "🏗️ Selecting prefabs...")
                    
                    if self.alpha_prefabs_path:
                        generator = RequirementsDrivenGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_from_requirements(requirements=requirements)
                    else:
                        # Fallback to NLP
                        grid, _ = nlp.generate_base(description, width, height)
                
                elif mode == "nlp":
                    progress.next("Generating with NLP...")
                    update_progress(2, "💬 Generating with NLP...")
                    grid, _ = nlp.generate_base(description, width, height)
                
                elif mode == "prefab":
                    progress.next("Placing prefabs...")
                    update_progress(2, "🏗️ Placing prefabs...")
                    
                    if self.alpha_prefabs_path:
                        from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
                        generator = HybridPrefabGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_with_prefab_anchors()
                
                elif mode == "hybrid":
                    progress.next("Hybrid generation...")
                    update_progress(2, "🔀 Hybrid generation...")
                    
                    if self.alpha_prefabs_path:
                        generator = EnhancedHybridGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_enhanced()
                
                elif mode == "ai":
                    progress.next("Consulting AI designer...")
                    update_progress(2, "🤖 Consulting AI designer...")
                    
                    if os.getenv("ANTHROPIC_API_KEY"):
                        designer = ClaudeBaseDesigner()
                        request = BaseDesignRequest(
                            colonist_count=requirements.num_colonists,
                            map_size=(width, height),
                            difficulty=requirements.defense_level
                        )
                        plan = designer.design_base(request)
                        
                        if plan and self.alpha_prefabs_path:
                            generator = RequirementsDrivenGenerator(width, height, self.alpha_prefabs_path)
                            grid = generator.generate_from_requirements(design_plan=plan)
                    else:
                        # Fallback
                        grid, _ = nlp.generate_base(description, width, height)
                
                progress.next("Finalizing...")
                update_progress(7, "✨ Finalizing...")
                
                if grid is not None:
                    self.last_grid = grid
                    
                    # Visualize
                    progress.next("Creating visualization...")
                    update_progress(8, "🎨 Creating visualization...")
                    
                    self.root.after(0, lambda: self.display_generation(grid))
                    self.root.after(0, lambda: self.update_status("✓ Generation complete!", "success"))
                else:
                    self.root.after(0, lambda: self.update_status("Generation failed", "error"))
                
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error: {str(e)}", "error"))
            finally:
                self.root.after(0, self.on_generation_complete)
        
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
    
    def display_generation(self, grid):
        """Display generated base"""
        self.current_view = "generated"  # Set to generated view
        
        # Use the layered visualizer without flip (we fixed it in the map loading instead)
        img = self.visualizer.visualize(grid, title="Generated Base", 
                                       show_legend=True, flip_y=False)
        img.save("generated_base.png")
        
        # Create smaller version for display
        display_size = (600, 600)
        display_img = img.resize(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_img)
        
        # Clear canvas and display
        self.map_canvas.delete("all")
        self.map_canvas.create_image(300, 300, image=photo)
        self.map_canvas.image = photo  # Keep reference
        
        self.selection_label.config(text="Base generated successfully! Layer controls are now active.", fg=ModernTheme.SUCCESS)
    
    def on_generation_complete(self):
        """Clean up after generation"""
        self.generate_btn.config(state=tk.NORMAL)
        self.progress_frame.pack_forget()
    
    def export_base(self):
        """Export generated base"""
        if self.last_grid is None:
            messagebox.showwarning("No Base", "Generate a base first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Base",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JSON Data", "*.json"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            if file_path.endswith('.json'):
                # Export as JSON
                data = {
                    "grid": self.last_grid.tolist(),
                    "width": self.last_grid.shape[1],
                    "height": self.last_grid.shape[0],
                    "generated": datetime.now().isoformat()
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                # Export as PNG with high quality using layered visualizer
                img = self.visualizer.visualize(self.last_grid, 
                                              title=f"RimWorld Base {self.base_width.get()}×{self.base_height.get()}", 
                                              show_legend=True, flip_y=False)
                img.save(file_path, quality=95)
            
            self.update_status(f"Exported to {Path(file_path).name}")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """RimWorld Base Assistant - Help

GENERATION MODES:
• Smart Generate: Best option, uses NLP + prefabs
• Natural Language: Describe base in plain English
• Prefab Anchors: Uses complete room designs
• Enhanced Hybrid: Combines multiple techniques
• AI Designer: Uses Claude AI for planning

TIPS:
• Be specific in your requirements
• Use Realistic Rooms dimensions for compact bases
• Include defense requirements for killboxes
• Mention colonist count for proper sizing
• Specify workshop types needed

BEST PRACTICES:
• Place bedrooms on edges for quiet
• Keep kitchen near freezer
• Hospital near entrance
• Workshops clustered together
• Central recreation room

KEYBOARD SHORTCUTS:
• Ctrl+L: Load save file
• Ctrl+G: Generate base
• Ctrl+E: Export base
• F1: Show this help"""
        
        messagebox.showinfo("Help", help_text)
    
    def update_status(self, message, level="info"):
        """Update status bar with color coding"""
        colors = {
            "info": ModernTheme.TEXT_LIGHT,
            "success": "#2ECC71",
            "error": "#E74C3C",
            "warning": "#F39C12"
        }
        
        prefix = {
            "success": "✓ ",
            "error": "✗ ",
            "warning": "⚠ ",
            "info": ""
        }
        
        self.status_bar.config(
            text=prefix.get(level, "") + message,
            fg=colors.get(level, ModernTheme.TEXT_LIGHT)
        )
        self.root.update()
    
    def run(self):
        """Run the application"""
        # Bind keyboard shortcuts
        self.root.bind('<Control-l>', lambda e: self.load_save())
        self.root.bind('<Control-g>', lambda e: self.generate_base())
        self.root.bind('<Control-e>', lambda e: self.export_base())
        self.root.bind('<F1>', lambda e: self.show_help())
        
        # Start
        self.root.mainloop()


if __name__ == "__main__":
    app = RimWorldAssistantGUI()
    app.run()
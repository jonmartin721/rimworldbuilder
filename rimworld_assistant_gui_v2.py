#!/usr/bin/env python3
"""
RimWorld Base Assistant - Enhanced GUI Interface
Modern, user-friendly interface with detailed mode descriptions and visual improvements.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Canvas
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import json
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.save_parser import RimWorldSaveParser
from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator, PrefabUsageMode
from src.generators.requirements_driven_generator import RequirementsDrivenGenerator
from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.ai.claude_base_designer import ClaudeBaseDesigner, BaseDesignRequest
from src.utils.progress import spinner
from src.utils.symbols import SUCCESS, FAILURE, WARNING


class ModernButton(tk.Button):
    """Modern styled button"""
    def __init__(self, parent, text="", command=None, style="primary", **kwargs):
        colors = {
            "primary": {"bg": "#007ACC", "fg": "white", "hover": "#005A9E"},
            "success": {"bg": "#28A745", "fg": "white", "hover": "#218838"},
            "danger": {"bg": "#DC3545", "fg": "white", "hover": "#C82333"},
            "secondary": {"bg": "#6C757D", "fg": "white", "hover": "#5A6268"}
        }
        
        color = colors.get(style, colors["primary"])
        
        super().__init__(
            parent, 
            text=text, 
            command=command,
            bg=color["bg"],
            fg=color["fg"],
            font=("Arial", 10),
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2",
            **kwargs
        )
        
        self.hover_color = color["hover"]
        self.default_color = color["bg"]
        
        self.bind("<Enter>", lambda e: self.config(bg=self.hover_color))
        self.bind("<Leave>", lambda e: self.config(bg=self.default_color))


class GenerationModeCard(tk.Frame):
    """Card widget for generation mode selection"""
    def __init__(self, parent, title, description, icon="🏗️", command=None):
        super().__init__(parent, bg="white", relief=tk.RAISED, bd=1)
        
        self.command = command
        
        # Card content
        padding = 15
        
        # Icon and title
        header = tk.Frame(self, bg="white")
        header.pack(fill=tk.X, padx=padding, pady=(padding, 5))
        
        tk.Label(header, text=icon, font=("Arial", 20), bg="white").pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(header, text=title, font=("Arial", 12, "bold"), bg="white").pack(side=tk.LEFT)
        
        # Description
        desc_label = tk.Label(
            self, 
            text=description, 
            font=("Arial", 9),
            bg="white",
            fg="#666666",
            wraplength=250,
            justify=tk.LEFT
        )
        desc_label.pack(fill=tk.X, padx=padding, pady=(0, padding))
        
        # Hover effect
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        
        for child in self.winfo_children():
            child.bind("<Enter>", self._on_enter)
            child.bind("<Leave>", self._on_leave)
            child.bind("<Button-1>", self._on_click)
    
    def _on_enter(self, event):
        self.config(bg="#F0F8FF")
        for child in self.winfo_children():
            if isinstance(child, (tk.Label, tk.Frame)):
                child.config(bg="#F0F8FF")
    
    def _on_leave(self, event):
        self.config(bg="white")
        for child in self.winfo_children():
            if isinstance(child, (tk.Label, tk.Frame)):
                child.config(bg="white")
    
    def _on_click(self, event):
        if self.command:
            self.command()


class MapCanvas(Canvas):
    """Interactive canvas for selecting base location on map"""
    def __init__(self, parent, width=500, height=500):
        super().__init__(parent, width=width, height=height, bg="#2A2A2A", highlightthickness=0)
        
        self.map_image = None
        self.selection_rect = None
        self.start_x = None
        self.start_y = None
        self.selected_area = None
        
        # Bind mouse events
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        
        # Draw grid
        self.draw_grid()
    
    def draw_grid(self):
        """Draw a grid on the canvas"""
        grid_size = 20
        for i in range(0, 500, grid_size):
            self.create_line(i, 0, i, 500, fill="#333333", width=1)
            self.create_line(0, i, 500, i, fill="#333333", width=1)
    
    def on_click(self, event):
        """Start selection"""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.selection_rect:
            self.delete(self.selection_rect)
        
        self.selection_rect = self.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="#00FF00", width=2, dash=(5, 5)
        )
    
    def on_drag(self, event):
        """Update selection rectangle"""
        if self.selection_rect:
            self.coords(self.selection_rect, self.start_x, self.start_y, event.x, event.y)
    
    def on_release(self, event):
        """Finish selection"""
        if self.start_x and self.start_y:
            x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
            x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
            
            # Convert to grid coordinates (assuming 250x250 map)
            map_scale = 250 / 500  # Map size / canvas size
            self.selected_area = (
                int(x1 * map_scale),
                int(y1 * map_scale),
                int(x2 * map_scale),
                int(y2 * map_scale)
            )
            
            # Update rectangle to show final selection
            if self.selection_rect:
                self.itemconfig(self.selection_rect, outline="#00FFFF", width=3)
    
    def load_map(self, game_state):
        """Load and display map from game state"""
        if game_state and game_state.maps:
            # Create a simple visualization of the map
            first_map = game_state.maps[0]
            
            # Create image
            img = Image.new('RGB', (500, 500), color='#1A1A1A')
            draw = ImageDraw.Draw(img)
            
            # Draw buildings as dots
            scale = 500 / 250  # Canvas size / map size
            for building in first_map.buildings:
                x = int(building.position.x * scale)
                y = int(building.position.y * scale)
                
                # Color based on building type
                if "Wall" in building.def_name:
                    color = "#666666"
                elif "Door" in building.def_name:
                    color = "#8B4513"
                elif "Bed" in building.def_name:
                    color = "#4169E1"
                elif "Table" in building.def_name:
                    color = "#8B7355"
                elif "Bridge" in building.def_name:
                    color = "#00CED1"
                else:
                    color = "#808080"
                
                draw.rectangle([x-1, y-1, x+1, y+1], fill=color)
            
            # Convert to PhotoImage
            self.map_image = ImageTk.PhotoImage(img)
            self.create_image(0, 0, anchor=tk.NW, image=self.map_image)
            
            # Redraw grid on top
            self.draw_grid()
            
            # Redraw selection if exists
            if self.selection_rect:
                self.tag_raise(self.selection_rect)


class RimWorldAssistantGUI:
    """Enhanced GUI for RimWorld Base Assistant"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RimWorld Base Assistant - Enhanced")
        self.root.geometry("1400x900")
        self.root.configure(bg="#F5F5F5")
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Variables
        self.save_path = None
        self.game_state = None
        self.last_grid = None
        self.current_image = None
        
        # Generation settings
        self.base_width = tk.IntVar(value=60)
        self.base_height = tk.IntVar(value=60)
        self.generation_mode = tk.StringVar(value="smart")
        
        # Check for AlphaPrefabs
        self.alpha_prefabs_path = Path("data/AlphaPrefabs")
        if not self.alpha_prefabs_path.exists():
            self.alpha_prefabs_path = None
        
        self.create_widgets()
        self.update_status("Welcome to RimWorld Base Assistant!")
    
    def create_widgets(self):
        """Create all GUI widgets with modern styling"""
        
        # Top toolbar
        toolbar = tk.Frame(self.root, bg="#2C3E50", height=50)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)
        
        # Logo/Title
        title_label = tk.Label(
            toolbar,
            text="🏰 RimWorld Base Assistant",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#2C3E50"
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Toolbar buttons
        ModernButton(toolbar, "Load Save", self.load_save, "primary").pack(side=tk.LEFT, padx=5)
        ModernButton(toolbar, "Export", self.export_base, "secondary").pack(side=tk.LEFT, padx=5)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#F5F5F5")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Generation modes
        left_panel = tk.Frame(main_container, bg="#F5F5F5", width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Section title
        tk.Label(
            left_panel,
            text="Generation Modes",
            font=("Arial", 14, "bold"),
            bg="#F5F5F5"
        ).pack(pady=(0, 15))
        
        # Mode cards
        modes = [
            {
                "title": "Smart Generate",
                "desc": "Best option! Uses NLP to understand your requirements, then intelligently selects and places real RimWorld prefabs.",
                "icon": "🎯",
                "mode": "smart"
            },
            {
                "title": "Natural Language",
                "desc": "Describe your base in plain English. Uses Wave Function Collapse for procedural generation.",
                "icon": "💬",
                "mode": "nlp"
            },
            {
                "title": "Prefab Anchors",
                "desc": "Places complete prefab designs as anchors, then fills remaining space procedurally.",
                "icon": "🏗️",
                "mode": "prefab"
            },
            {
                "title": "Enhanced Hybrid",
                "desc": "Uses multiple prefab modes (complete, partial, decorative) for the most varied output.",
                "icon": "🔀",
                "mode": "hybrid"
            },
            {
                "title": "AI Designer",
                "desc": "Uses Claude AI to create detailed specifications, then generates accordingly.",
                "icon": "🤖",
                "mode": "ai"
            }
        ]
        
        for mode in modes:
            card = GenerationModeCard(
                left_panel,
                mode["title"],
                mode["desc"],
                mode["icon"],
                lambda m=mode["mode"]: self.select_mode(m)
            )
            card.pack(fill=tk.X, pady=5)
        
        # Middle panel - Input and settings
        middle_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=1)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Input section
        input_frame = tk.Frame(middle_panel, bg="white")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(
            input_frame,
            text="Describe Your Base",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor=tk.W)
        
        tk.Label(
            input_frame,
            text="Examples: 'defensive base for 8 colonists with medical bay and killbox'",
            font=("Arial", 9),
            fg="#666666",
            bg="white"
        ).pack(anchor=tk.W, pady=(5, 10))
        
        # Larger text input
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=6,
            font=("Arial", 11),
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=1,
            highlightbackground="#CCCCCC",
            highlightthickness=1
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.insert("1.0", "Create a defensive base for 10 colonists with:\n- Medical bay near entrance\n- Central dining hall\n- Individual bedrooms\n- Workshop area\n- Killbox defense")
        
        # Size controls
        size_frame = tk.Frame(middle_panel, bg="white")
        size_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(
            size_frame,
            text="Base Dimensions",
            font=("Arial", 11, "bold"),
            bg="white"
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        
        tk.Label(size_frame, text="Width:", bg="white").grid(row=1, column=0, sticky=tk.W)
        width_spin = tk.Spinbox(
            size_frame,
            from_=20, to=200,
            textvariable=self.base_width,
            width=10,
            font=("Arial", 10)
        )
        width_spin.grid(row=1, column=1, padx=(5, 20))
        
        tk.Label(size_frame, text="Height:", bg="white").grid(row=1, column=2, sticky=tk.W)
        height_spin = tk.Spinbox(
            size_frame,
            from_=20, to=200,
            textvariable=self.base_height,
            width=10,
            font=("Arial", 10)
        )
        height_spin.grid(row=1, column=3, padx=5)
        
        # Preset sizes
        tk.Label(size_frame, text="Presets:", bg="white").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        presets = [
            ("Small (40x40)", 40, 40),
            ("Medium (60x60)", 60, 60),
            ("Large (80x80)", 80, 80),
            ("Huge (100x100)", 100, 100)
        ]
        
        preset_frame = tk.Frame(size_frame, bg="white")
        preset_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        for text, w, h in presets:
            tk.Button(
                preset_frame,
                text=text,
                command=lambda width=w, height=h: self.set_size(width, height),
                font=("Arial", 9),
                relief=tk.FLAT,
                bg="#E0E0E0",
                padx=10,
                pady=2
            ).pack(side=tk.LEFT, padx=2)
        
        # Generate button
        ModernButton(
            middle_panel,
            "🚀 Generate Base",
            self.generate_base,
            "success",
            font=("Arial", 12, "bold")
        ).pack(pady=20)
        
        # Right panel - Map/Preview
        right_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_panel,
            text="Map Selection",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(pady=10)
        
        tk.Label(
            right_panel,
            text="Click and drag to select base location",
            font=("Arial", 9),
            fg="#666666",
            bg="white"
        ).pack()
        
        # Map canvas
        self.map_canvas = MapCanvas(right_panel)
        self.map_canvas.pack(padx=20, pady=10)
        
        # Selected area info
        self.area_label = tk.Label(
            right_panel,
            text="No area selected",
            font=("Arial", 10),
            bg="white",
            fg="#666666"
        )
        self.area_label.pack(pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 9),
            bg="#34495E",
            fg="white",
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_mode(self, mode):
        """Select generation mode"""
        self.generation_mode.set(mode)
        self.update_status(f"Selected mode: {mode}")
    
    def set_size(self, width, height):
        """Set base dimensions"""
        self.base_width.set(width)
        self.base_height.set(height)
        self.update_status(f"Base size set to {width}x{height}")
    
    def load_save(self):
        """Load a RimWorld save file"""
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
                    
                    # Update map canvas
                    self.root.after(0, lambda: self.map_canvas.load_map(self.game_state))
                    
                    if self.game_state and self.game_state.maps:
                        first_map = self.game_state.maps[0]
                        self.update_status(f"Loaded: {self.save_path.name} - {len(first_map.buildings)} buildings")
                    else:
                        self.update_status("Failed to parse save file")
                        
                except Exception as e:
                    self.update_status(f"Error: {str(e)}")
            
            # Run in thread to avoid freezing UI
            thread = threading.Thread(target=load)
            thread.daemon = True
            thread.start()
    
    def generate_base(self):
        """Generate base based on selected mode"""
        mode = self.generation_mode.get()
        description = self.text_input.get("1.0", tk.END).strip()
        width = self.base_width.get()
        height = self.base_height.get()
        
        if not description:
            messagebox.showwarning("Input Required", "Please describe your base requirements")
            return
        
        self.update_status(f"Generating {width}x{height} base using {mode} mode...")
        
        def generate():
            try:
                grid = None
                
                if mode == "smart":
                    # Smart generation with requirements-driven generator
                    if not self.alpha_prefabs_path:
                        self.update_status("AlphaPrefabs not found - using fallback generation")
                        mode = "nlp"
                    else:
                        nlp = BaseGeneratorNLP()
                        requirements = nlp.parse_request(description)
                        
                        generator = RequirementsDrivenGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_from_requirements(requirements=requirements)
                
                if mode == "nlp" or (mode == "smart" and grid is None):
                    # Natural language generation
                    nlp = BaseGeneratorNLP()
                    grid, _ = nlp.generate_base(description, width, height)
                
                elif mode == "prefab":
                    # Prefab anchor generation
                    if self.alpha_prefabs_path:
                        from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
                        generator = HybridPrefabGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_with_prefab_anchors(
                            prefab_categories=["bedroom", "kitchen", "workshop"],
                            num_prefabs=4,
                            fill_with_wfc=True
                        )
                
                elif mode == "hybrid":
                    # Enhanced hybrid generation
                    if self.alpha_prefabs_path:
                        generator = EnhancedHybridGenerator(width, height, self.alpha_prefabs_path)
                        grid = generator.generate_enhanced(
                            usage_modes=[PrefabUsageMode.COMPLETE, PrefabUsageMode.PARTIAL],
                            prefab_density=0.4
                        )
                
                elif mode == "ai":
                    # AI-designed generation
                    if os.getenv("ANTHROPIC_API_KEY"):
                        designer = ClaudeBaseDesigner()
                        nlp = BaseGeneratorNLP()
                        requirements = nlp.parse_request(description)
                        
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
                        self.update_status("Claude API key not set - using NLP fallback")
                        nlp = BaseGeneratorNLP()
                        grid, _ = nlp.generate_base(description, width, height)
                
                if grid is not None:
                    self.last_grid = grid
                    self.save_and_display_grid(grid)
                    self.update_status(f"Generation complete! Base saved as generated.png")
                else:
                    self.update_status("Generation failed - please try again")
                    
            except Exception as e:
                self.update_status(f"Error during generation: {str(e)}")
        
        # Run in thread
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
    
    def save_and_display_grid(self, grid):
        """Save grid as image and display preview"""
        from src.generators.wfc_generator import TileType
        
        # Create image
        scale = 5
        height, width = grid.shape
        img = Image.new('RGB', (width * scale, height * scale), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        # Color mapping - fixed to ensure visibility
        colors = {
            0: (50, 50, 50),      # Empty - dark gray
            1: (150, 150, 150),   # Wall - light gray
            2: (100, 80, 60),     # Floor - brown
            3: (120, 100, 80),    # Room - tan
            4: (139, 69, 19),     # Door - dark brown
            5: (70, 130, 180),    # Furniture - steel blue
            6: (255, 215, 0),     # Storage - gold
            7: (50, 205, 50),     # Power - green
            8: (255, 140, 0)      # Other - orange
        }
        
        # Draw tiles
        for y in range(height):
            for x in range(width):
                tile = int(grid[y, x])
                color = colors.get(tile, (100, 100, 100))
                draw.rectangle(
                    [x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1],
                    fill=color
                )
        
        # Save
        img.save("generated.png")
        
        # TODO: Display preview in GUI
    
    def export_base(self):
        """Export generated base"""
        if self.last_grid is None:
            messagebox.showwarning("No Base", "Generate a base first before exporting")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Base",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JSON Data", "*.json")]
        )
        
        if file_path:
            if file_path.endswith('.json'):
                # Export as JSON
                data = {
                    "grid": self.last_grid.tolist(),
                    "width": self.last_grid.shape[1],
                    "height": self.last_grid.shape[0]
                }
                with open(file_path, 'w') as f:
                    json.dump(data, f)
            else:
                # Already saved as PNG
                import shutil
                shutil.copy("generated.png", file_path)
            
            self.update_status(f"Exported to {Path(file_path).name}")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    app = RimWorldAssistantGUI()
    app.run()
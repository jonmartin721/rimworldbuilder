#!/usr/bin/env python3
"""
RimWorld Base Assistant - GUI Interface
Graphical interface for the RimWorld Base Assistant using tkinter.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageTk
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.save_parser import RimWorldSaveParser
from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator, PrefabUsageMode
from src.nlp.base_generator_nlp import BaseGeneratorNLP


class RimWorldAssistantGUI:
    """GUI interface for RimWorld Base Assistant"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RimWorld Base Assistant")
        self.root.geometry("1200x800")
        
        # Variables
        self.save_path = None
        self.game_state = None
        self.last_grid = None
        self.current_image = None
        
        # Check for AlphaPrefabs
        self.alpha_prefabs_path = Path("data/AlphaPrefabs")
        if not self.alpha_prefabs_path.exists():
            self.alpha_prefabs_path = None
        
        self.create_widgets()
        self.update_status("Welcome to RimWorld Base Assistant!")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Save File", command=self.load_save)
        file_menu.add_command(label="Export Base", command=self.export_base)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        generate_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Generate", menu=generate_menu)
        generate_menu.add_command(label="Natural Language", command=self.show_nlp_dialog)
        generate_menu.add_command(label="Prefab Anchors", command=self.show_prefab_dialog)
        generate_menu.add_command(label="Enhanced Hybrid", command=self.show_hybrid_dialog)
        
        # Main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Save file info
        ttk.Label(left_panel, text="Save File:").grid(row=0, column=0, sticky=tk.W)
        self.save_label = ttk.Label(left_panel, text="None loaded")
        self.save_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Button(left_panel, text="Load Save", command=self.load_save).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Quick generate section
        ttk.Separator(left_panel, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_panel, text="Quick Generate:", font=('Arial', 10, 'bold')).grid(row=3, column=0, columnspan=2)
        
        # NLP input
        ttk.Label(left_panel, text="Describe base:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.nlp_entry = ttk.Entry(left_panel, width=30)
        self.nlp_entry.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.nlp_entry.insert(0, "Efficient base for 6 colonists")
        
        ttk.Button(left_panel, text="Generate from Text", 
                  command=self.quick_nlp_generate).grid(row=6, column=0, columnspan=2, pady=5)
        
        # Generation options
        ttk.Separator(left_panel, orient='horizontal').grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_panel, text="Generation Options:", font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2)
        
        # Size selection
        ttk.Label(left_panel, text="Base size:").grid(row=9, column=0, sticky=tk.W)
        self.size_var = tk.StringVar(value="60x60")
        size_combo = ttk.Combobox(left_panel, textvariable=self.size_var, width=10)
        size_combo['values'] = ('40x40', '60x60', '80x80', '100x100')
        size_combo.grid(row=9, column=1, sticky=tk.W)
        
        # Prefab density
        ttk.Label(left_panel, text="Prefab density:").grid(row=10, column=0, sticky=tk.W)
        self.density_var = tk.DoubleVar(value=0.4)
        density_scale = ttk.Scale(left_panel, from_=0.1, to=0.9, 
                                 variable=self.density_var, orient=tk.HORIZONTAL)
        density_scale.grid(row=10, column=1, sticky=(tk.W, tk.E))
        
        # Mode selection
        ttk.Label(left_panel, text="Mode:").grid(row=11, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="hybrid")
        mode_combo = ttk.Combobox(left_panel, textvariable=self.mode_var, width=15)
        mode_combo['values'] = ('simple', 'prefabs', 'hybrid', 'enhanced')
        mode_combo.grid(row=11, column=1, sticky=tk.W)
        
        ttk.Button(left_panel, text="Generate Base", 
                  command=self.generate_base).grid(row=12, column=0, columnspan=2, pady=10)
        
        # Stats section
        ttk.Separator(left_panel, orient='horizontal').grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_panel, text="Statistics:", font=('Arial', 10, 'bold')).grid(row=14, column=0, columnspan=2)
        
        self.stats_text = tk.Text(left_panel, width=30, height=10, wrap=tk.WORD)
        self.stats_text.grid(row=15, column=0, columnspan=2, pady=5)
        
        # Center panel - Visualization
        center_panel = ttk.LabelFrame(main_frame, text="Base Visualization", padding="10")
        center_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for image
        self.canvas = tk.Canvas(center_panel, width=600, height=600, bg='black')
        self.canvas.pack()
        
        # Right panel - Log
        right_panel = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        right_panel.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(right_panel, width=40, height=40, wrap=tk.WORD)
        self.log_text.pack()
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def load_save(self):
        """Load a save file"""
        filename = filedialog.askopenfilename(
            title="Select RimWorld Save File",
            filetypes=[("RimWorld Saves", "*.rws"), ("All Files", "*.*")]
        )
        
        if filename:
            self.save_path = Path(filename)
            self.log(f"Loading {self.save_path.name}...")
            
            # Load in thread to avoid freezing
            def load():
                try:
                    parser = RimWorldSaveParser()
                    self.game_state = parser.parse(str(self.save_path))
                    
                    if self.game_state and self.game_state.maps:
                        first_map = self.game_state.maps[0]
                        self.save_label.config(text=self.save_path.name)
                        
                        stats = f"Map size: {first_map.size}\n"
                        stats += f"Buildings: {len(first_map.buildings)}\n"
                        stats += f"Colonists: {len(first_map.get_colonists())}\n"
                        
                        bridges = [b for b in first_map.buildings if "Bridge" in b.def_name]
                        if bridges:
                            stats += f"Bridges: {len(bridges)}\n"
                        
                        self.update_stats(stats)
                        self.log("Save loaded successfully!")
                        self.update_status(f"Loaded: {self.save_path.name}")
                    else:
                        self.log("Error: Could not parse save file")
                except Exception as e:
                    self.log(f"Error loading save: {e}")
            
            threading.Thread(target=load, daemon=True).start()
    
    def quick_nlp_generate(self):
        """Quick generation from NLP input"""
        description = self.nlp_entry.get()
        if not description:
            messagebox.showwarning("Input Required", "Please enter a base description")
            return
        
        self.log(f"Generating: {description}")
        
        # Get size
        size_str = self.size_var.get()
        width, height = map(int, size_str.split('x'))
        
        # Generate in thread
        def generate():
            try:
                nlp = BaseGeneratorNLP()
                grid, desc = nlp.generate_base(description, width, height)
                self.last_grid = grid
                
                self.log("Generation complete!")
                self.log(desc)
                self.display_grid(grid)
                self.update_grid_stats(grid)
                self.update_status("Base generated successfully")
            except Exception as e:
                self.log(f"Error: {e}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def generate_base(self):
        """Generate base with selected options"""
        mode = self.mode_var.get()
        size_str = self.size_var.get()
        width, height = map(int, size_str.split('x'))
        density = self.density_var.get()
        
        self.log(f"Generating {mode} base ({width}x{height}, density: {density:.1%})...")
        
        def generate():
            try:
                if mode == 'hybrid' or mode == 'enhanced':
                    if not self.alpha_prefabs_path:
                        self.log("AlphaPrefabs not found! Using simple generation.")
                        mode = 'simple'
                    else:
                        gen = EnhancedHybridGenerator(width, height, self.alpha_prefabs_path)
                        grid = gen.generate_enhanced(
                            usage_modes=[PrefabUsageMode.COMPLETE, PrefabUsageMode.PARTIAL],
                            prefab_density=density
                        )
                        self.last_grid = grid
                        
                elif mode == 'prefabs':
                    from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
                    gen = HybridPrefabGenerator(width, height, self.alpha_prefabs_path)
                    grid = gen.generate_with_prefab_anchors(num_prefabs=int(density * 10))
                    self.last_grid = grid
                    
                else:  # simple
                    from src.generators.wfc_generator import WFCGenerator
                    gen = WFCGenerator(width, height)
                    gen.generate()
                    # Convert to grid
                    grid = np.zeros((height, width), dtype=int)
                    for y in range(height):
                        for x in range(width):
                            tile = gen.get_tile(x, y)
                            if tile and tile.final_type:
                                grid[y, x] = tile.final_type.value
                    self.last_grid = grid
                
                self.log("Generation complete!")
                self.display_grid(self.last_grid)
                self.update_grid_stats(self.last_grid)
                self.update_status(f"{mode.capitalize()} base generated")
                
            except Exception as e:
                self.log(f"Error: {e}")
        
        threading.Thread(target=generate, daemon=True).start()
    
    def display_grid(self, grid):
        """Display grid on canvas"""
        # Create image from grid
        from scripts.test_hybrid_generation import visualize_grid
        
        # Save to temp file
        temp_path = "temp_visualization.png"
        visualize_grid(grid, temp_path)
        
        # Load and display
        img = Image.open(temp_path)
        # Resize to fit canvas
        img = img.resize((600, 600), Image.Resampling.NEAREST)
        
        # Convert to PhotoImage
        self.current_image = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(300, 300, image=self.current_image)
    
    def update_grid_stats(self, grid):
        """Update statistics for generated grid"""
        from src.generators.wfc_generator import TileType
        
        stats = f"Size: {grid.shape[1]}x{grid.shape[0]}\n\n"
        
        unique, counts = np.unique(grid, return_counts=True)
        for val, count in zip(unique, counts):
            try:
                tile_type = TileType(val)
                percent = count / grid.size * 100
                stats += f"{tile_type.name}: {count} ({percent:.1f}%)\n"
            except:
                pass
        
        self.update_stats(stats)
    
    def export_base(self):
        """Export the generated base"""
        if self.last_grid is None:
            messagebox.showwarning("No Base", "Please generate a base first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Base",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("CSV File", "*.csv"), ("All Files", "*.*")]
        )
        
        if filename:
            if filename.endswith('.csv'):
                np.savetxt(filename, self.last_grid, delimiter=',', fmt='%d')
            else:
                from scripts.test_hybrid_generation import visualize_grid
                visualize_grid(self.last_grid, filename)
            
            self.log(f"Exported to {Path(filename).name}")
            self.update_status(f"Exported: {Path(filename).name}")
    
    def show_nlp_dialog(self):
        """Show NLP generation dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Natural Language Generation")
        dialog.geometry("500x400")
        
        ttk.Label(dialog, text="Describe your ideal base:", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        text = tk.Text(dialog, width=60, height=10, wrap=tk.WORD)
        text.pack(padx=20, pady=10)
        text.insert('1.0', "Create a defensive base for 8 colonists with:\n"
                           "- Central killbox entrance\n"
                           "- Hospital near entrance\n"
                           "- Workshops clustered for efficiency\n"
                           "- Individual bedrooms\n"
                           "- Large storage area")
        
        def generate():
            description = text.get('1.0', tk.END).strip()
            if description:
                self.nlp_entry.delete(0, tk.END)
                self.nlp_entry.insert(0, description[:100])
                dialog.destroy()
                self.quick_nlp_generate()
        
        ttk.Button(dialog, text="Generate", command=generate).pack(pady=10)
    
    def show_prefab_dialog(self):
        """Show prefab selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Prefab Anchor Generation")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Select Prefab Categories:", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        categories = ['bedroom', 'kitchen', 'storage', 'workshop', 
                     'medical', 'recreation', 'power']
        
        vars = {}
        for cat in categories:
            var = tk.BooleanVar(value=cat in ['bedroom', 'kitchen', 'workshop'])
            vars[cat] = var
            ttk.Checkbutton(dialog, text=cat.capitalize(), 
                          variable=var).pack(anchor=tk.W, padx=20)
        
        def generate():
            selected = [cat for cat, var in vars.items() if var.get()]
            if selected:
                dialog.destroy()
                self.log(f"Generating with prefabs: {', '.join(selected)}")
                self.generate_base()
        
        ttk.Button(dialog, text="Generate", command=generate).pack(pady=20)
    
    def show_hybrid_dialog(self):
        """Show enhanced hybrid options dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enhanced Hybrid Generation")
        dialog.geometry("400x350")
        
        ttk.Label(dialog, text="Usage Modes:", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        modes = {
            'Complete Prefabs': PrefabUsageMode.COMPLETE,
            'Partial Rooms': PrefabUsageMode.PARTIAL,
            'Decorative Elements': PrefabUsageMode.DECORATIVE,
            'Conceptual Patterns': PrefabUsageMode.CONCEPTUAL
        }
        
        vars = {}
        for name, mode in modes.items():
            var = tk.BooleanVar(value=mode in [PrefabUsageMode.COMPLETE, 
                                               PrefabUsageMode.PARTIAL])
            vars[mode] = var
            ttk.Checkbutton(dialog, text=name, variable=var).pack(anchor=tk.W, padx=20)
        
        ttk.Separator(dialog, orient='horizontal').pack(fill=tk.X, pady=20)
        
        ttk.Label(dialog, text="Density:").pack()
        density_var = tk.DoubleVar(value=0.4)
        ttk.Scale(dialog, from_=0.1, to=0.9, variable=density_var, 
                 orient=tk.HORIZONTAL, length=200).pack()
        
        def generate():
            selected_modes = [mode for mode, var in vars.items() if var.get()]
            if selected_modes:
                self.density_var.set(density_var.get())
                dialog.destroy()
                self.generate_base()
        
        ttk.Button(dialog, text="Generate", command=generate).pack(pady=20)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
    
    def update_stats(self, stats):
        """Update statistics display"""
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = RimWorldAssistantGUI()
    app.run()


if __name__ == "__main__":
    main()
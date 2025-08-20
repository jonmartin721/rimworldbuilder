"""
Interactive visualization with layer toggles for RimWorld base layouts.
Uses matplotlib for interactive display with checkboxes to toggle layers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import matplotlib.patches as patches
from src.parser.save_parser import RimWorldSaveParser as SaveParser
from src.models.game_entities import BuildingType
from collections import defaultdict


class InteractiveVisualizer:
    """Interactive visualization with layer toggles"""
    
    def __init__(self, save_path: str):
        """Initialize with save file path"""
        self.parser = SaveParser()
        self.parse_result = self.parser.parse(save_path)
        self.layers = self._organize_layers()
        self.layer_visibility = {layer: True for layer in self.layers}
        self.setup_plot()
        
    def _organize_layers(self):
        """Organize buildings into layers by type"""
        layers = defaultdict(list)
        
        for building in self.parse_result.buildings:
            layers[building.building_type].append(building)
        
        # Sort layers by number of buildings (most common first)
        sorted_layers = sorted(layers.keys(), 
                             key=lambda x: len(layers[x]), 
                             reverse=True)
        
        self.layer_data = {layer: layers[layer] for layer in sorted_layers}
        return sorted_layers
    
    def setup_plot(self):
        """Set up the matplotlib plot with interactive controls"""
        # Create figure with main plot and control panel
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main plot area
        self.ax_main = plt.subplot2grid((1, 5), (0, 0), colspan=4)
        self.ax_main.set_title(f"RimWorld Base Layout - {len(self.parse_result.buildings)} buildings")
        self.ax_main.set_xlabel("X Coordinate")
        self.ax_main.set_ylabel("Y Coordinate")
        
        # Set map bounds
        if self.parse_result.map_size:
            self.ax_main.set_xlim(0, self.parse_result.map_size[0])
            self.ax_main.set_ylim(0, self.parse_result.map_size[1])
            self.ax_main.invert_yaxis()  # Invert Y axis for proper orientation
        
        # Control panel for layer toggles
        self.ax_control = plt.subplot2grid((1, 5), (0, 4))
        self.ax_control.set_title("Layer Controls")
        
        # Create checkboxes for each layer
        labels = []
        for layer in self.layers:
            count = len(self.layer_data[layer])
            label = f"{layer.name} ({count})"
            labels.append(label)
        
        # Position checkboxes
        y_positions = np.linspace(0.9, 0.1, len(labels))
        self.check = CheckButtons(self.ax_control, labels, 
                                 [True] * len(labels))
        
        # Connect checkbox callback
        self.check.on_clicked(self.toggle_layer)
        
        # Initial draw
        self.draw_layers()
        
        # Add grid
        self.ax_main.grid(True, alpha=0.3)
        
        # Add legend with colors
        self.add_legend()
        
        plt.tight_layout()
    
    def get_layer_color(self, building_type: BuildingType):
        """Get color for a building type"""
        colors = {
            BuildingType.WALL: '#808080',
            BuildingType.DOOR: '#8B4513',
            BuildingType.BEDROOM: '#6495ED',
            BuildingType.KITCHEN: '#FF8C00',
            BuildingType.STORAGE: '#32CD32',
            BuildingType.RECREATION: '#DDA0DD',
            BuildingType.WORKSHOP: '#FFD700',
            BuildingType.MEDICAL: '#FF6B6B',
            BuildingType.RESEARCH: '#9370DB',
            BuildingType.POWER: '#FFFF00',
            BuildingType.BRIDGE: '#4682B4',
            BuildingType.FLOOR: '#D2691E',
            BuildingType.CONDUIT: '#FFA500',
            BuildingType.FENCE: '#8B7355',
            BuildingType.LIGHT: '#FFFACD',
            BuildingType.OTHER: '#C0C0C0'
        }
        return colors.get(building_type, '#C0C0C0')
    
    def get_marker_style(self, building_type: BuildingType):
        """Get marker style for a building type"""
        markers = {
            BuildingType.WALL: 's',  # square
            BuildingType.DOOR: 'P',  # plus (filled)
            BuildingType.BEDROOM: 'o',  # circle
            BuildingType.KITCHEN: '^',  # triangle up
            BuildingType.STORAGE: 'D',  # diamond
            BuildingType.RECREATION: '*',  # star
            BuildingType.WORKSHOP: 'h',  # hexagon
            BuildingType.MEDICAL: '+',  # plus
            BuildingType.RESEARCH: 'p',  # pentagon
            BuildingType.POWER: 'X',  # X (filled)
            BuildingType.BRIDGE: '_',  # horizontal line
            BuildingType.FLOOR: '.',  # point
            BuildingType.CONDUIT: '|',  # vertical line
            BuildingType.FENCE: '1',  # tri down
            BuildingType.LIGHT: 'v',  # triangle down
            BuildingType.OTHER: '.'   # point
        }
        return markers.get(building_type, '.')
    
    def draw_layers(self):
        """Draw all visible layers"""
        self.ax_main.clear()
        self.ax_main.set_title(f"RimWorld Base Layout - {len(self.parse_result.buildings)} buildings")
        self.ax_main.set_xlabel("X Coordinate")
        self.ax_main.set_ylabel("Y Coordinate")
        
        # Set map bounds
        if self.parse_result.map_size:
            self.ax_main.set_xlim(0, self.parse_result.map_size[0])
            self.ax_main.set_ylim(0, self.parse_result.map_size[1])
            self.ax_main.invert_yaxis()
        
        # Draw each visible layer
        for i, layer in enumerate(self.layers):
            if self.layer_visibility[layer]:
                buildings = self.layer_data[layer]
                
                # Extract positions
                x_coords = [b.position.x for b in buildings]
                y_coords = [b.position.y for b in buildings]
                
                # Plot with specific color and marker
                color = self.get_layer_color(layer)
                marker = self.get_marker_style(layer)
                
                # Adjust marker size based on building type
                marker_size = 3 if layer in [BuildingType.CONDUIT, BuildingType.FLOOR] else 5
                
                self.ax_main.scatter(x_coords, y_coords, 
                                    c=color, 
                                    marker=marker,
                                    s=marker_size,
                                    alpha=0.7,
                                    label=f"{layer.name} ({len(buildings)})")
        
        # Draw colonists if present
        if self.parse_result.colonists:
            colonist_x = [c.position.x for c in self.parse_result.colonists]
            colonist_y = [c.position.y for c in self.parse_result.colonists]
            self.ax_main.scatter(colonist_x, colonist_y, 
                               c='red', marker='*', s=100, 
                               label=f"Colonists ({len(self.parse_result.colonists)})",
                               zorder=10)
        
        # Add grid
        self.ax_main.grid(True, alpha=0.3)
        
        # Update display
        self.fig.canvas.draw_idle()
    
    def toggle_layer(self, label):
        """Toggle layer visibility when checkbox is clicked"""
        # Extract layer type from label (remove count)
        layer_name = label.split(' (')[0]
        
        # Find matching BuildingType
        for layer in self.layers:
            if layer.name == layer_name:
                self.layer_visibility[layer] = not self.layer_visibility[layer]
                break
        
        # Redraw
        self.draw_layers()
    
    def add_legend(self):
        """Add a color legend for building types"""
        # Create custom legend
        legend_elements = []
        for layer in self.layers[:10]:  # Show top 10 types
            if self.layer_visibility[layer]:
                color = self.get_layer_color(layer)
                count = len(self.layer_data[layer])
                element = patches.Patch(color=color, 
                                      label=f"{layer.name} ({count})")
                legend_elements.append(element)
        
        if legend_elements:
            self.ax_main.legend(handles=legend_elements, 
                              loc='upper right',
                              fontsize=8,
                              framealpha=0.8)
    
    def show(self):
        """Display the interactive plot"""
        plt.show()


def main():
    """Main function to run interactive visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive RimWorld base visualization")
    parser.add_argument("save_file", nargs="?", 
                       default="data/saves/Autosave-2.rws",
                       help="Path to RimWorld save file")
    
    args = parser.parse_args()
    
    save_path = Path(args.save_file)
    if not save_path.exists():
        print(f"Save file not found: {save_path}")
        # Try default location
        save_path = Path("data/saves/Autosave-2.rws")
        if not save_path.exists():
            print("No save file found. Please provide a valid .rws file.")
            return
    
    print(f"Loading save file: {save_path}")
    print("Creating interactive visualization...")
    print("Use checkboxes to toggle layers on/off")
    
    visualizer = InteractiveVisualizer(str(save_path))
    visualizer.show()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
RimWorld Base Assistant - Main Interface
A comprehensive tool for analyzing RimWorld saves and generating optimized base layouts.
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
from typing import Optional, List
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parser.save_parser import RimWorldSaveParser
from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator, PrefabUsageMode
from src.generators.requirements_driven_generator import RequirementsDrivenGenerator
from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.ai.claude_base_designer import ClaudeBaseDesigner, BaseDesignRequest
from src.utils.progress import spinner, StepProgress, log_section, log_item, GenerationLogger
from src.utils.symbols import SUCCESS, FAILURE, WARNING, FOLDER, CHART, PENCIL, ROBOT, HAMMER


class RimWorldAssistant:
    """Main interface for the RimWorld Base Assistant"""
    
    def __init__(self):
        self.save_path = None
        self.game_state = None
        self.generator = None
        self.nlp = None
        self.ai_designer = None
        self.last_grid = None
        
        # Check for AlphaPrefabs
        self.alpha_prefabs_path = Path("data/AlphaPrefabs")
        if not self.alpha_prefabs_path.exists():
            print("Warning: AlphaPrefabs mod not found. Some features will be limited.")
            print("To get full functionality, clone: https://github.com/juanosarg/AlphaPrefabs")
            self.alpha_prefabs_path = None
    
    def run(self):
        """Run the interactive interface"""
        self.print_header()
        
        while True:
            try:
                self.print_menu()
                choice = input("\nEnter your choice: ").strip()
                
                if choice == '1':
                    self.load_save_file()
                elif choice == '2':
                    self.analyze_current_base()
                elif choice == '3':
                    self.generate_with_nlp()
                elif choice == '4':
                    self.generate_with_prefabs()
                elif choice == '5':
                    self.generate_hybrid()
                elif choice == '6':
                    self.ai_design_base()
                elif choice == '7':
                    self.smart_generate()
                elif choice == '8':
                    self.visualize_last()
                elif choice == '9':
                    self.interactive_viewer()
                elif choice == '*':
                    self.export_base()
                elif choice == '0' or choice.lower() == 'q':
                    print("\nThank you for using RimWorld Base Assistant!")
                    break
                else:
                    print("Invalid choice. Please try again.")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again or choose a different option.")
    
    def print_header(self):
        """Print the application header"""
        print("=" * 70)
        print("                    RIMWORLD BASE ASSISTANT")
        print("=" * 70)
        print("AI-powered base design using real prefabs and procedural generation")
        print("-" * 70)
    
    def print_menu(self):
        """Print the main menu"""
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        
        if self.save_path:
            print(f"Current save: {self.save_path.name}")
        else:
            print("No save file loaded")
        
        if self.last_grid is not None:
            print(f"Last generation: {self.last_grid.shape[0]}x{self.last_grid.shape[1]} tiles")
        
        print("\n1. Load Save File")
        print("2. Analyze Current Base")
        print("3. Generate from Natural Language")
        print("4. Generate with Prefab Anchors")
        print("5. Generate Enhanced Hybrid Base")
        print("6. AI-Designed Base (Claude)")
        print("7. Smart Generate (NLP → Prefabs)")
        print("8. Visualize Last Generation")
        print("9. Interactive Layer Viewer")
        print("*. Export Base Design")
        print("0. Exit")
    
    def load_save_file(self):
        """Load a RimWorld save file"""
        log_section("LOAD SAVE FILE", 50)
        
        # List available saves
        saves_dir = Path("data/saves")
        if saves_dir.exists():
            saves = list(saves_dir.glob("*.rws"))
            if saves:
                print("\nAvailable saves:")
                for i, save in enumerate(saves, 1):
                    print(f"  {i}. {save.name}")
                print(f"  {len(saves)+1}. Enter custom path")
                
                choice = input("\nSelect save (number or path): ").strip()
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(saves):
                        self.save_path = saves[idx]
                    elif idx == len(saves):
                        custom = input("Enter save file path: ").strip()
                        self.save_path = Path(custom)
                except:
                    self.save_path = Path(choice)
        else:
            path = input("Enter save file path: ").strip()
            self.save_path = Path(path)
        
        # Parse the save with progress indicator
        if self.save_path and self.save_path.exists():
            print(f"\n{FOLDER} Loading {self.save_path.name}...")
            
            with spinner("Parsing save file"):
                parser = RimWorldSaveParser()
                self.game_state = parser.parse(str(self.save_path))
            
            if self.game_state and self.game_state.maps:
                first_map = self.game_state.maps[0]
                print(f"{SUCCESS} Loaded successfully!")
                log_item("Map size", str(first_map.size))
                log_item("Buildings", str(len(first_map.buildings)))
                log_item("Colonists", str(len(first_map.get_colonists())))
                
                # Count bridges
                bridges = [b for b in first_map.buildings if "Bridge" in b.def_name]
                if bridges:
                    log_item("Buildable bridges", str(len(bridges)))
            else:
                print(f"{FAILURE} Error: Could not parse save file")
        else:
            print(f"{FAILURE} Error: Save file not found")
    
    def analyze_current_base(self):
        """Analyze the current base from save file"""
        print("\n" + "-" * 40)
        print("BASE ANALYSIS")
        print("-" * 40)
        
        if not self.game_state:
            print("Please load a save file first!")
            return
        
        first_map = self.game_state.maps[0]
        
        # Analyze buildings by type
        from collections import Counter
        building_types = Counter()
        for building in first_map.buildings:
            building_types[building.building_type.name] += 1
        
        print("\nBuilding Distribution:")
        for btype, count in building_types.most_common(10):
            print(f"  {btype}: {count}")
        
        # Analyze rooms
        print("\nRoom Analysis:")
        room_types = {
            "Bedroom": ["Bed", "Bedroll"],
            "Kitchen": ["Stove", "Kitchen"],
            "Storage": ["Shelf", "Storage"],
            "Workshop": ["TableMachining", "TableStonecutter", "Bench"],
            "Recreation": ["Television", "ChessTable", "BilliardsTable"],
            "Hospital": ["HospitalBed", "VitalMonitor"]
        }
        
        for room_type, keywords in room_types.items():
            count = sum(1 for b in first_map.buildings 
                       if any(kw in b.def_name for kw in keywords))
            if count > 0:
                print(f"  {room_type}: {count} furniture items")
        
        # Find buildable areas
        bridges = [b for b in first_map.buildings if "Bridge" in b.def_name]
        if bridges:
            x_coords = [b.position.x for b in bridges]
            y_coords = [b.position.y for b in bridges]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            print(f"\nBuildable bridge area: {max_x-min_x+1}x{max_y-min_y+1} tiles")
            print(f"  Coverage: {len(bridges)} tiles ({len(bridges)/((max_x-min_x+1)*(max_y-min_y+1))*100:.1f}%)")
    
    def generate_with_nlp(self):
        """Generate base from natural language description"""
        print("\n" + "-" * 40)
        print("NATURAL LANGUAGE GENERATION")
        print("-" * 40)
        
        print("\nDescribe your ideal base:")
        print("Examples:")
        print("  - 'Create a defensive base for 8 colonists with killbox'")
        print("  - 'Build a compact base for 5 people with workshop and hospital'")
        print("  - 'Make an efficient base with 6 bedrooms and large storage'")
        
        description = input("\nYour description: ").strip()
        if not description:
            return
        
        # Get dimensions
        size = input("Base size (default 60x60): ").strip()
        if size:
            try:
                width, height = map(int, size.split('x'))
            except:
                width, height = 60, 60
        else:
            width, height = 60, 60
        
        print(f"\nGenerating {width}x{height} base...")
        
        # Initialize NLP if needed
        if not self.nlp:
            self.nlp = BaseGeneratorNLP(str(self.save_path) if self.save_path else None)
        
        # Generate
        grid, description_result = self.nlp.generate_base(description, width, height)
        self.last_grid = grid
        
        print("\n" + description_result)
        
        # Save visualization
        self.save_visualization(grid, "nlp_generated.png")
        print(f"\nVisualization saved to nlp_generated.png")
    
    def generate_with_prefabs(self):
        """Generate using real prefab anchors"""
        print("\n" + "-" * 40)
        print("PREFAB ANCHOR GENERATION")
        print("-" * 40)
        
        if not self.alpha_prefabs_path:
            print("AlphaPrefabs mod not found! Cannot use real prefabs.")
            return
        
        # Get parameters
        print("\nSelect prefab categories (comma-separated):")
        print("Available: bedroom, kitchen, storage, workshop, medical, recreation, power")
        categories = input("Categories (default: bedroom,kitchen,workshop): ").strip()
        if not categories:
            categories = "bedroom,kitchen,workshop"
        category_list = [c.strip() for c in categories.split(',')]
        
        num_prefabs = input("Number of prefabs to place (default 4): ").strip()
        num_prefabs = int(num_prefabs) if num_prefabs else 4
        
        size = input("Base size (default 60x60): ").strip()
        if size:
            try:
                width, height = map(int, size.split('x'))
            except:
                width, height = 60, 60
        else:
            width, height = 60, 60
        
        print(f"\nGenerating {width}x{height} base with {num_prefabs} prefabs...")
        
        # Create generator
        from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
        generator = HybridPrefabGenerator(width, height, self.alpha_prefabs_path)
        
        # Generate
        grid = generator.generate_with_prefab_anchors(
            prefab_categories=category_list,
            num_prefabs=num_prefabs,
            fill_with_wfc=True
        )
        self.last_grid = grid
        
        print(f"\nPlaced {len(generator.placed_prefabs)} prefabs:")
        for prefab in generator.placed_prefabs:
            print(f"  - {prefab.layout.def_name} at {prefab.position}")
        
        # Save visualization
        self.save_visualization(grid, "prefab_anchored.png")
        print(f"\nVisualization saved to prefab_anchored.png")
    
    def generate_hybrid(self):
        """Generate enhanced hybrid base"""
        print("\n" + "-" * 40)
        print("ENHANCED HYBRID GENERATION")
        print("-" * 40)
        
        if not self.alpha_prefabs_path:
            print("AlphaPrefabs mod not found! Limited functionality.")
            return
        
        print("\nUsage modes:")
        print("1. Complete prefabs only")
        print("2. Complete + Partial rooms")
        print("3. Complete + Partial + Decorative")
        print("4. All modes (most varied)")
        
        mode_choice = input("Select mode (1-4, default 3): ").strip()
        
        modes_map = {
            '1': [PrefabUsageMode.COMPLETE],
            '2': [PrefabUsageMode.COMPLETE, PrefabUsageMode.PARTIAL],
            '3': [PrefabUsageMode.COMPLETE, PrefabUsageMode.PARTIAL, PrefabUsageMode.DECORATIVE],
            '4': [PrefabUsageMode.COMPLETE, PrefabUsageMode.PARTIAL, 
                  PrefabUsageMode.DECORATIVE, PrefabUsageMode.CONCEPTUAL]
        }
        modes = modes_map.get(mode_choice, modes_map['3'])
        
        density = input("Prefab density (0.1-0.9, default 0.4): ").strip()
        density = float(density) if density else 0.4
        
        size = input("Base size (default 80x80): ").strip()
        if size:
            try:
                width, height = map(int, size.split('x'))
            except:
                width, height = 80, 80
        else:
            width, height = 80, 80
        
        print(f"\nGenerating enhanced {width}x{height} base...")
        
        # Create generator
        generator = EnhancedHybridGenerator(width, height, self.alpha_prefabs_path)
        
        # Generate
        grid = generator.generate_enhanced(
            usage_modes=modes,
            prefab_density=density,
            decoration_density=0.2
        )
        self.last_grid = grid
        
        mode_names = [m.value for m in modes]
        print(f"\nGenerated with modes: {', '.join(mode_names)}")
        print(f"Prefab density: {density:.0%}")
        
        # Save visualization
        self.save_visualization(grid, "enhanced_hybrid.png")
        print(f"\nVisualization saved to enhanced_hybrid.png")
    
    def smart_generate(self):
        """Smart generation that combines NLP understanding with intelligent prefab selection"""
        print("\n" + "-" * 40)
        print("SMART GENERATION (NLP → PREFABS)")
        print("-" * 40)
        
        if not self.alpha_prefabs_path:
            print("AlphaPrefabs mod not found! This feature requires AlphaPrefabs.")
            print("Please clone: https://github.com/juanosarg/AlphaPrefabs into data/AlphaPrefabs")
            return
        
        print("\nThis mode uses natural language to understand your requirements,")
        print("then intelligently selects and places real RimWorld prefabs.")
        
        print(f"\n{PENCIL} Describe your ideal base:")
        print("Examples:")
        print("  - 'Defensive base for 8 colonists with medical bay and killbox'")
        print("  - 'Efficient production base with 4 workshops and large storage'")
        print("  - 'Comfortable base for 10 colonists with recreation and dining'")
        
        description = input("\n> Your requirements: ").strip()
        if not description:
            return
        
        # Parse with NLP
        print("\n⏳ Processing your requirements...")
        
        with spinner("Parsing natural language"):
            if not self.nlp:
                self.nlp = BaseGeneratorNLP(str(self.save_path) if self.save_path else None)
            
            requirements = self.nlp.parse_request(description)
        
        print(f"\n{CHART} Parsed Requirements:")
        log_item("Colonists", str(requirements.num_colonists))
        log_item("Bedrooms", str(requirements.num_bedrooms))
        log_item("Style", requirements.style)
        log_item("Defense", requirements.defense_level)
        
        # Optionally use Claude for detailed planning
        use_ai = input(f"\n{ROBOT} Use Claude AI for detailed planning? (y/n, default n): ").lower() == 'y'
        
        plan = None
        if use_ai and os.getenv("ANTHROPIC_API_KEY"):
            print("Requesting AI design...")
            if not self.ai_designer:
                self.ai_designer = ClaudeBaseDesigner()
            
            request = BaseDesignRequest(
                colonist_count=requirements.num_colonists,
                map_size=(60, 60),
                difficulty=requirements.defense_level,
                priorities=[requirements.style],
                special_requirements=requirements.special_requirements or []
            )
            plan = self.ai_designer.design_base(request)
            
            if plan:
                print(f"\n✨ AI Strategy: {plan.layout_strategy}")
                print(f"🛡️ Defense: {plan.defense_strategy}")
        
        # Generate with requirements-driven generator
        print(f"\n{HAMMER} Generating base with intelligent prefab selection...")
        
        generator = RequirementsDrivenGenerator(60, 60, self.alpha_prefabs_path)
        grid = generator.generate_from_requirements(
            requirements=requirements,
            design_plan=plan
        )
        self.last_grid = grid
        
        # Save visualization
        self.save_visualization(grid, "smart_generated.png")
        print(f"\n{SUCCESS} Smart-generated base saved to smart_generated.png")
        print("This base uses real RimWorld prefabs matched to your requirements!")
    
    def ai_design_base(self):
        """Use Claude AI to design base"""
        print("\n" + "-" * 40)
        print("AI-DESIGNED BASE (Claude)")
        print("-" * 40)
        
        # Check for API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\nNote: No Claude API key found. Using fallback design.")
            print("To use Claude AI, set environment variable ANTHROPIC_API_KEY")
        
        # Get parameters
        colonists = input("\nNumber of colonists (default 6): ").strip()
        colonists = int(colonists) if colonists else 6
        
        print("\nDifficulty level:")
        print("1. Peaceful")
        print("2. Medium")
        print("3. Hard")
        print("4. Extreme")
        
        diff_choice = input("Select (1-4, default 2): ").strip()
        difficulty_map = {'1': 'peaceful', '2': 'medium', '3': 'hard', '4': 'extreme'}
        difficulty = difficulty_map.get(diff_choice, 'medium')
        
        priorities = input("Priorities (e.g., defense,efficiency,comfort): ").strip()
        priority_list = [p.strip() for p in priorities.split(',')] if priorities else ['balanced']
        
        special = input("Special requirements (e.g., killbox,hospital,throne_room): ").strip()
        special_list = [s.strip() for s in special.split(',')] if special else []
        
        print("\nRequesting AI design...")
        
        # Initialize designer
        if not self.ai_designer:
            self.ai_designer = ClaudeBaseDesigner()
        
        # Create request
        request = BaseDesignRequest(
            colonist_count=colonists,
            map_size=(60, 60),
            difficulty=difficulty,
            priorities=priority_list,
            special_requirements=special_list
        )
        
        # Get design
        plan = self.ai_designer.design_base(request)
        
        if plan:
            print(f"\n=== AI BASE DESIGN ===")
            print(f"Strategy: {plan.layout_strategy}")
            print(f"Defense: {plan.defense_strategy}")
            print(f"Traffic: {plan.traffic_flow}")
            
            print("\nRoom Layout:")
            for spec in plan.room_specs[:10]:
                print(f"  {spec.quantity}x {spec.room_type}: {spec.size[0]}x{spec.size[1]} tiles")
            
            print(f"\nResources needed: {plan.estimated_resources}")
            
            # Option to generate
            if input("\nGenerate this base? (y/n): ").lower() == 'y':
                self.generate_from_ai_plan(plan)
    
    def generate_from_ai_plan(self, plan):
        """Generate base from AI plan using requirements-driven generator"""
        if not self.alpha_prefabs_path:
            print("Cannot generate without AlphaPrefabs mod")
            return
        
        print("\nGenerating base from AI design plan...")
        
        # Use the new requirements-driven generator
        generator = RequirementsDrivenGenerator(60, 60, self.alpha_prefabs_path)
        
        # Generate using the design plan
        grid = generator.generate_from_requirements(design_plan=plan)
        self.last_grid = grid
        
        self.save_visualization(grid, "ai_designed.png")
        print(f"\nAI-designed base saved to ai_designed.png")
    
    def visualize_last(self):
        """Visualize the last generated base"""
        if self.last_grid is None:
            print("\nNo base has been generated yet!")
            return
        
        print("\nVisualizing last generation...")
        self.save_visualization(self.last_grid, "last_generation.png")
        print("Saved to last_generation.png")
        
        # Show stats
        from src.generators.wfc_generator import TileType
        unique, counts = np.unique(self.last_grid, return_counts=True)
        print(f"\nBase composition ({self.last_grid.shape[0]}x{self.last_grid.shape[1]} tiles):")
        for val, count in zip(unique, counts):
            try:
                tile_type = TileType(val)
                percent = count / self.last_grid.size * 100
                print(f"  {tile_type.name}: {count} tiles ({percent:.1f}%)")
            except:
                pass
    
    def interactive_viewer(self):
        """Launch interactive layer viewer"""
        if not self.save_path or not self.game_state:
            print("\nPlease load a save file first!")
            return
        
        print("\nLaunching interactive viewer...")
        print("Use checkboxes to toggle layers on/off")
        
        # Import and run
        from scripts.interactive_visualize import InteractiveVisualizer
        visualizer = InteractiveVisualizer(str(self.save_path))
        visualizer.show()
    
    def export_base(self):
        """Export base design"""
        if self.last_grid is None:
            print("\nNo base to export!")
            return
        
        print("\n" + "-" * 40)
        print("EXPORT BASE DESIGN")
        print("-" * 40)
        
        print("Export formats:")
        print("1. PNG image")
        print("2. CSV grid")
        print("3. JSON data")
        
        format_choice = input("Select format (1-3): ").strip()
        
        if format_choice == '1':
            filename = input("Filename (default: exported_base.png): ").strip()
            filename = filename if filename else "exported_base.png"
            self.save_visualization(self.last_grid, filename)
            print(f"Exported to {filename}")
            
        elif format_choice == '2':
            filename = input("Filename (default: exported_base.csv): ").strip()
            filename = filename if filename else "exported_base.csv"
            np.savetxt(filename, self.last_grid, delimiter=',', fmt='%d')
            print(f"Exported to {filename}")
            
        elif format_choice == '3':
            import json
            filename = input("Filename (default: exported_base.json): ").strip()
            filename = filename if filename else "exported_base.json"
            data = {
                'width': self.last_grid.shape[1],
                'height': self.last_grid.shape[0],
                'grid': self.last_grid.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Exported to {filename}")
    
    def save_visualization(self, grid, filename):
        """Save grid as image"""
        with spinner(f"Saving visualization to {filename}"):
            from src.visualization import BaseVisualizer
            visualizer = BaseVisualizer(scale=10, show_grid=True)
            visualizer.visualize(grid, filename, title="Generated RimWorld Base", show_legend=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RimWorld Base Assistant')
    parser.add_argument('--save', help='Path to save file to load')
    parser.add_argument('--generate', help='Generate type: nlp, prefab, hybrid, ai')
    parser.add_argument('--request', help='Natural language request for generation')
    parser.add_argument('--output', help='Output filename', default='generated_base.png')
    
    args = parser.parse_args()
    
    assistant = RimWorldAssistant()
    
    # Handle command-line arguments for non-interactive mode
    if args.save or args.generate:
        if args.save:
            assistant.save_path = Path(args.save)
            if assistant.save_path.exists():
                parser = RimWorldSaveParser()
                assistant.game_state = parser.parse(str(assistant.save_path))
                print(f"Loaded save: {assistant.save_path}")
        
        if args.generate:
            if args.generate == 'nlp' and args.request:
                assistant.nlp = BaseGeneratorNLP()
                grid, desc = assistant.nlp.generate_base(args.request, 60, 60)
                assistant.save_visualization(grid, args.output)
                print(f"Generated base saved to {args.output}")
                
            elif args.generate == 'prefab':
                from src.generators.hybrid_prefab_generator import HybridPrefabGenerator
                gen = HybridPrefabGenerator(60, 60, assistant.alpha_prefabs_path)
                grid = gen.generate_with_prefab_anchors()
                assistant.save_visualization(grid, args.output)
                print(f"Generated base saved to {args.output}")
                
            else:
                print(f"Unknown generation type: {args.generate}")
    else:
        # Interactive mode
        assistant.run()


if __name__ == "__main__":
    main()
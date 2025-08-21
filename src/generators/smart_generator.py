"""
Smart unified generator that automatically handles everything:
- Analyzes terrain to find buildable areas
- Understands user requirements from natural language
- Uses AlphaPrefabs as inspiration/anchors
- Applies RimWorld best practices
- Generates optimal base design
"""


from src.generators.zone_aware_generator import ZoneAwareGenerator
from src.analysis.terrain_analyzer import TerrainAnalyzer
from src.nlp.base_generator_nlp import BaseGeneratorNLP
from src.ai.claude_base_designer import ClaudeBaseDesigner
from src.generators.requirements_driven_generator import RequirementsDrivenGenerator
from src.models.base_grid import BaseGrid


class SmartGenerator:
    """
    One generator to rule them all - handles everything automatically

    CLEAN SLATE APPROACH:
    - Treats ALL existing structures as replaceable
    - Only respects terrain (water/mountains)
    - Automatically determines best approach based on context
    """

    def __init__(self, game_state=None):
        self.game_state = game_state
        self.terrain_analyzer = TerrainAnalyzer()
        self.nlp_parser = BaseGeneratorNLP()
        self.zone_generator = ZoneAwareGenerator(game_state)
        self.requirements_generator = RequirementsDrivenGenerator()

        # RimWorld best practices
        self.best_practices = {
            "room_sizes": {
                "bedroom": (4, 4),  # Minimum for decent mood
                "bedroom_good": (6, 5),  # Impressive bedroom
                "dining": (11, 11),  # Combined rec/dining
                "kitchen": (7, 6),  # With space for multiple stoves
                "freezer": (7, 7),  # Adjacent to kitchen
                "workshop": (13, 13),  # Multiple workbenches
                "hospital": (8, 6),  # 4-6 medical beds
                "prison": (4, 4),  # Per cell
                "storage": (11, 11),  # Standard stockpile
                "battery": (5, 3),  # Battery room
                "hydroponics": (13, 5),  # Sunlamp coverage
            },
            "adjacency": {
                "kitchen": ["freezer", "dining"],
                "hospital": ["storage"],  # For medicine
                "workshop": ["storage"],
                "bedrooms": ["dining", "recreation"],
            },
            "defense": {
                "killbox_size": (15, 25),
                "turret_spacing": 8,
                "wall_thickness": 2,  # For important areas
                "sandbag_coverage": True,
            },
            "efficiency": {
                "central_heating": True,
                "power_grid": "redundant",
                "traffic_flow": "minimize_crossing",
                "separate_clean_dirty": True,
            },
        }

    def generate(
        self,
        user_request: str,
        width: int = 100,
        height: int = 100,
        use_claude: bool = False,
    ) -> dict:
        """
        Generate a complete base design from user request

        Args:
            user_request: Natural language description of desired base
            width: Base width (will auto-adjust based on terrain)
            height: Base height (will auto-adjust based on terrain)
            use_claude: Whether to use Claude AI for advanced planning

        Returns:
            Dict with:
                - grid: The generated base layout
                - plan: Detailed plan of what was built
                - terrain_analysis: Information about the terrain
                - requirements: Parsed requirements
                - zones: Zone assignments
        """

        print("🤖 Smart Generator starting...")

        # Step 1: Parse user requirements
        print("📝 Understanding your requirements...")
        requirements = self.nlp_parser.parse_request(user_request)

        # Step 2: Analyze terrain
        print("🗺️ Analyzing terrain...")
        terrain_analysis = None
        if self.game_state:
            foundation_grid = getattr(self.game_state, "foundation_grid", None)
            terrain_analysis = self.terrain_analyzer.analyze_terrain(
                self.game_state, foundation_grid
            )

            # Auto-adjust size based on available space
            if terrain_analysis and terrain_analysis["buildable_areas"]:
                largest_area = terrain_analysis["buildable_areas"][0]
                width = min(width, largest_area.width)
                height = min(height, largest_area.height)
                print(f"  Adjusted to fit largest buildable area: {width}x{height}")

        # Step 3: Create detailed plan (optionally with Claude)
        print("📐 Creating detailed base plan...")
        detailed_plan = None

        if use_claude:
            try:
                designer = ClaudeBaseDesigner()
                # Convert requirements to Claude format
                claude_request = self._requirements_to_claude_format(
                    requirements, user_request
                )
                detailed_plan = designer.design_base(claude_request)
                print("  ✅ Claude AI provided advanced design")
            except Exception as e:
                print(f"  ⚠️ Claude unavailable, using local planning: {e}")

        if not detailed_plan:
            # Use local planning with best practices
            detailed_plan = self._create_local_plan(requirements, terrain_analysis)
            print("  ✅ Created plan using RimWorld best practices")

        # Step 4: Determine zone usage
        print("🎯 Planning zone usage...")
        zone_preferences = {
            "use_outer_areas": requirements.use_outer_areas,
            "agriculture_zones": requirements.agriculture_zones,
            "defensive_layers": requirements.defensive_layers,
            "bridge_everything": requirements.bridge_water,
            "ignore_existing": True,  # Always true - clean slate
        }

        # Step 5: Generate the base
        print("🔨 Generating base layout...")

        # Use requirements-driven generator with prefab matching
        self.requirements_generator.prefab_analyzer = (
            self.zone_generator.prefab_analyzer
        )

        # Convert detailed plan to requirements format
        generation_requirements = self._plan_to_requirements(
            detailed_plan, requirements
        )

        # Generate with zone awareness if we have terrain
        if terrain_analysis and requirements.use_outer_areas:
            grid = self.zone_generator.generate_with_zones(
                width, height, generation_requirements, zone_preferences
            )
            print("  ✅ Generated multi-zone base")
        else:
            # Standard generation
            grid = self.requirements_generator.generate_from_requirements(
                generation_requirements, width, height
            )
            print("  ✅ Generated base layout")

        # Step 6: Apply best practices post-processing
        print("⚡ Applying RimWorld best practices...")
        self._apply_best_practices(grid, requirements)

        # Step 7: Create detailed report
        print("📊 Creating detailed report...")
        report = self._create_report(
            grid, detailed_plan, terrain_analysis, requirements, zone_preferences
        )

        print("✨ Generation complete!")

        return {
            "grid": grid,
            "plan": detailed_plan,
            "terrain_analysis": terrain_analysis,
            "requirements": requirements,
            "zones": self.zone_generator.zone_assignments
            if hasattr(self.zone_generator, "zone_assignments")
            else {},
            "report": report,
        }

    def _requirements_to_claude_format(
        self, requirements, original_request: str
    ) -> dict:
        """Convert parsed requirements to Claude API format"""
        return {
            "description": original_request,
            "colonists": requirements.num_colonists,
            "priorities": [requirements.style, requirements.defense_level],
            "must_have": [
                room
                for room in [
                    "kitchen",
                    "dining",
                    "rec",
                    "storage",
                    "medical",
                    "research",
                    "prison",
                ]
                if getattr(requirements, f"include_{room}", False)
            ],
            "special": requirements.special_requirements or [],
        }

    def _create_local_plan(self, requirements, terrain_analysis) -> dict:
        """Create a detailed plan using local best practices"""
        plan = {"rooms": [], "defense": {}, "infrastructure": {}, "zones": {}}

        # Plan bedrooms
        for i in range(requirements.num_bedrooms):
            room_quality = (
                "bedroom_good" if requirements.style == "spacious" else "bedroom"
            )
            size = self.best_practices["room_sizes"][room_quality]
            plan["rooms"].append(
                {
                    "type": "bedroom",
                    "size": size,
                    "priority": 1,
                    "features": ["bed", "dresser", "end_table"]
                    if requirements.style == "spacious"
                    else ["bed"],
                }
            )

        # Plan common areas
        if requirements.include_dining:
            plan["rooms"].append(
                {
                    "type": "dining_hall",
                    "size": self.best_practices["room_sizes"]["dining"],
                    "priority": 1,
                    "features": ["tables", "chairs", "decorations"],
                    "adjacent_to": ["kitchen"],
                }
            )

        if requirements.include_kitchen:
            plan["rooms"].append(
                {
                    "type": "kitchen",
                    "size": self.best_practices["room_sizes"]["kitchen"],
                    "priority": 1,
                    "features": ["stoves", "butcher_table"],
                    "adjacent_to": ["freezer", "dining_hall"],
                }
            )

            # Always add freezer with kitchen
            plan["rooms"].append(
                {
                    "type": "freezer",
                    "size": self.best_practices["room_sizes"]["freezer"],
                    "priority": 1,
                    "features": ["coolers", "shelves"],
                }
            )

        # Plan workshops
        for i in range(requirements.num_workrooms):
            plan["rooms"].append(
                {
                    "type": "workshop",
                    "size": self.best_practices["room_sizes"]["workshop"],
                    "priority": 2,
                    "features": ["workbenches", "tool_cabinets"],
                }
            )

        # Plan defense based on level
        if requirements.defense_level == "high":
            plan["defense"] = {
                "killbox": self.best_practices["defense"]["killbox_size"],
                "turrets": True,
                "wall_layers": requirements.defensive_layers,
                "bunkers": True,
            }
        elif requirements.defense_level == "medium":
            plan["defense"] = {"turrets": True, "wall_layers": 1, "sandbags": True}

        # Plan zones if using outer areas
        if requirements.use_outer_areas:
            plan["zones"] = {
                "inner": "residential and essential services",
                "outer": "agriculture and defense",
                "connections": "bridges where needed",
            }

        return plan

    def _plan_to_requirements(self, plan: dict, base_requirements) -> dict:
        """Convert detailed plan to generation requirements"""
        req = {
            "rooms": plan.get("rooms", []),
            "colonist_count": base_requirements.num_colonists,
            "defense_level": base_requirements.defense_level,
            "style": base_requirements.style,
        }

        # Add defense requirements
        if "defense" in plan:
            if "killbox" in plan["defense"]:
                req["killbox"] = plan["defense"]["killbox"]
            req["turrets"] = plan["defense"].get("turrets", False)
            req["wall_layers"] = plan["defense"].get("wall_layers", 1)

        return req

    def _apply_best_practices(self, grid: BaseGrid, requirements):
        """Apply RimWorld best practices to the generated base"""

        # Ensure minimum room sizes
        self._ensure_minimum_room_sizes(grid)

        # Add double walls for important areas if defensive
        if requirements.defense_level == "high":
            self._add_double_walls(grid)

        # Ensure proper traffic flow
        self._optimize_traffic_flow(grid)

        # Add power infrastructure
        self._add_power_grid(grid)

    def _ensure_minimum_room_sizes(self, grid: BaseGrid):
        """Ensure all rooms meet minimum size requirements"""
        # This would analyze the grid and expand rooms that are too small
        pass  # Simplified for now

    def _add_double_walls(self, grid: BaseGrid):
        """Add double walls for better defense"""
        # This would thicken walls in critical areas
        pass  # Simplified for now

    def _optimize_traffic_flow(self, grid: BaseGrid):
        """Optimize pathways to minimize crossing"""
        # This would analyze and improve corridors
        pass  # Simplified for now

    def _add_power_grid(self, grid: BaseGrid):
        """Add power conduits and generators"""
        # This would add a proper power grid
        pass  # Simplified for now

    def _create_report(
        self, grid, plan, terrain_analysis, requirements, zone_preferences
    ) -> str:
        """Create a detailed report of what was generated"""
        report = []
        report.append("🏗️ BASE GENERATION REPORT")
        report.append("=" * 50)

        # Terrain summary
        if terrain_analysis:
            stats = terrain_analysis["terrain_stats"]
            report.append("\n📍 TERRAIN ANALYSIS:")
            report.append(f"  • Total buildable: {stats['buildable_tiles']} tiles")
            report.append(f"  • Buildable areas: {stats['num_buildable_areas']}")
            report.append(f"  • Largest area: {stats['largest_area_size']} tiles")
            if requirements.use_outer_areas:
                report.append("  • Using outer zones: YES")

        # Requirements summary
        report.append("\n👥 REQUIREMENTS:")
        report.append(f"  • Colonists: {requirements.num_colonists}")
        report.append(f"  • Style: {requirements.style}")
        report.append(f"  • Defense: {requirements.defense_level}")
        if requirements.agriculture_zones:
            report.append("  • Agriculture zones: YES")

        # What was built
        if plan and "rooms" in plan:
            report.append("\n🏠 STRUCTURES PLANNED:")
            room_counts = {}
            for room in plan["rooms"]:
                room_type = room.get("type", "unknown")
                room_counts[room_type] = room_counts.get(room_type, 0) + 1

            for room_type, count in room_counts.items():
                report.append(f"  • {room_type}: {count}")

        # Defense summary
        if plan and "defense" in plan:
            report.append("\n🛡️ DEFENSE:")
            defense = plan["defense"]
            if "killbox" in defense:
                report.append(f"  • Killbox: {defense['killbox']}")
            if defense.get("turrets"):
                report.append("  • Turrets: YES")
            if "wall_layers" in defense:
                report.append(f"  • Wall layers: {defense['wall_layers']}")

        # Zone usage
        if zone_preferences["use_outer_areas"]:
            report.append("\n🗺️ ZONE USAGE:")
            report.append("  • Inner zone: Core base & essential services")
            report.append("  • Outer zones: Agriculture & perimeter defense")
            if zone_preferences["bridge_everything"]:
                report.append("  • Bridges: Will build over shallow water")

        report.append("\n" + "=" * 50)
        report.append("✅ Generation complete - ready to build!")

        return "\n".join(report)

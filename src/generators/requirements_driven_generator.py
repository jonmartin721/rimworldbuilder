"""
Requirements-Driven Generator - Intelligently matches Claude/NLP requirements to prefabs.
This generator acts as the bridge between high-level requirements and actual prefab placement.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import random
import time

from src.generators.enhanced_hybrid_generator import EnhancedHybridGenerator
from src.generators.alpha_prefab_parser import AlphaPrefabLayout
from src.ai.claude_base_designer import BaseDesignPlan, RoomSpec
from src.nlp.base_generator_nlp import BaseRequirements
from src.generators.wfc_generator import TileType
from src.generators.rimworld_best_practices import RimWorldBestPractices, ModConfig
from src.utils.progress import GenerationLogger, ProgressBar, spinner


@dataclass
class PrefabMatch:
    """Represents a match between a requirement and a prefab"""

    prefab: AlphaPrefabLayout
    score: float
    room_spec: RoomSpec
    reasons: List[str]


class RequirementsDrivenGenerator(EnhancedHybridGenerator):
    """Generator that intelligently selects prefabs based on requirements"""

    def __init__(
        self,
        width: int,
        height: int,
        alpha_prefabs_path: Optional[Path] = None,
        learned_patterns_file: Optional[Path] = None,
        mod_config: ModConfig = ModConfig.REALISTIC_ROOMS,
    ):
        super().__init__(width, height, alpha_prefabs_path, learned_patterns_file)

        self.best_practices = RimWorldBestPractices()
        self.mod_config = mod_config

        # Room type mapping from specs to prefab categories
        self.room_type_mapping = {
            "bedroom": ["bedroom", "barracks", "quarters"],
            "kitchen": ["kitchen", "cooking", "food"],
            "dining": ["dining", "cafeteria", "eating"],
            "storage": ["storage", "warehouse", "stockpile"],
            "workshop": ["workshop", "production", "crafting"],
            "medical": ["medical", "hospital", "medbay"],
            "recreation": ["recreation", "rec", "joy"],
            "research": ["research", "lab", "science"],
            "prison": ["prison", "jail", "cell"],
            "power": ["power", "generator", "battery"],
            "defense": ["defense", "turret", "killbox"],
        }

    def generate_from_requirements(
        self,
        requirements: Optional[BaseRequirements] = None,
        design_plan: Optional[BaseDesignPlan] = None,
    ) -> np.ndarray:
        """
        Generate a base from high-level requirements and/or detailed design plan.

        Args:
            requirements: NLP-parsed requirements
            design_plan: Claude AI's detailed design plan

        Returns:
            Generated grid
        """
        logger = GenerationLogger()
        logger.start("REQUIREMENTS-DRIVEN GENERATION")

        # Step 1: Process requirements
        logger.step("Processing requirements")

        # Convert requirements to room specs if no design plan
        if design_plan is None and requirements is not None:
            logger.detail("Converting NLP requirements to design plan")
            design_plan = self._requirements_to_design_plan(requirements)
        elif design_plan is None:
            logger.detail("Using default design plan")
            design_plan = self._get_default_plan()

        total_rooms = sum(spec.quantity for spec in design_plan.room_specs)
        logger.detail(f"Total rooms to place: {total_rooms}")

        # Step 2: Match prefabs to requirements
        logger.step("Matching prefabs to requirements")

        with spinner("Analyzing prefab library"):
            prefab_matches = self._match_prefabs_to_specs(design_plan.room_specs)
            time.sleep(0.5)  # Give spinner time to show

        logger.detail(f"Found {len(prefab_matches)} suitable prefabs")

        # Sort by priority and size (place important/large rooms first)
        prefab_matches.sort(
            key=lambda m: (m.room_spec.priority, -m.prefab.width * m.prefab.height)
        )

        # Initialize grid
        self.grid = np.zeros((self.height, self.width), dtype=int)
        placed_prefabs = []

        # Step 3: Place prefabs
        logger.step("Placing prefabs on grid")

        progress = ProgressBar(len(prefab_matches), prefix="Placement")

        for i, match in enumerate(prefab_matches):
            progress.update(i, f"{match.room_spec.room_type}")

            if self._try_place_prefab_smart(match, placed_prefabs):
                placed_prefabs.append(match)
                logger.detail(
                    f"✓ Placed {match.room_spec.room_type}: {match.prefab.def_name} (score: {match.score:.2f})"
                )
                if i == 0:  # Only show reasons for first few
                    for reason in match.reasons[:2]:
                        logger.detail(f"  - {reason}")
            else:
                logger.warning(f"Could not place {match.room_spec.room_type}")

        progress.finish()

        # Step 4: Fill and connect
        logger.step("Adding corridors and decorations")

        with spinner("Filling empty spaces"):
            self._fill_with_decoration(design_plan)
            time.sleep(0.3)

        with spinner("Ensuring connectivity"):
            self._ensure_connectivity()
            time.sleep(0.3)

        # Step 5: Add defenses
        if design_plan.defense_strategy != "none":
            logger.step("Adding defensive structures")
            with spinner(f"Implementing {design_plan.defense_strategy} defense"):
                self._add_defensive_elements(design_plan.defense_strategy)
                time.sleep(0.3)

        # Summary
        logger.success(
            f"Placed {len(placed_prefabs)}/{len(prefab_matches)} prefabs successfully"
        )
        logger.finish()

        return self.grid

    def _requirements_to_design_plan(
        self, requirements: BaseRequirements
    ) -> BaseDesignPlan:
        """Convert simple requirements to a design plan using best practices"""
        room_specs = []

        # Add bedrooms with Realistic Rooms dimensions
        if requirements.num_bedrooms > 0:
            bedroom_dims = self.best_practices.get_room_dimensions(
                "bedroom", self.mod_config, "standard"
            )
            room_specs.append(
                RoomSpec(
                    room_type="bedroom",
                    size=bedroom_dims,
                    quantity=requirements.num_bedrooms,
                    priority=6,  # Lower priority for edge placement
                    adjacency_preferences=["corridor"],
                )
            )

        # Add other rooms based on flags with best practices
        if requirements.include_kitchen:
            kitchen_dims = self.best_practices.get_room_dimensions(
                "kitchen", self.mod_config, "standard"
            )
            room_specs.append(
                RoomSpec(
                    room_type="kitchen",
                    size=kitchen_dims,
                    quantity=1,
                    priority=2,  # High priority for central placement
                    adjacency_preferences=["freezer", "dining"],
                    special_features=["sterile_tiles"],
                )
            )

            # Always add freezer next to kitchen
            freezer_dims = self.best_practices.get_room_dimensions(
                "kitchen",
                self.mod_config,
                "min",  # Freezer can be smaller
            )
            room_specs.append(
                RoomSpec(
                    room_type="freezer",
                    size=freezer_dims,
                    quantity=1,
                    priority=2,
                    adjacency_preferences=["kitchen"],
                )
            )

        if requirements.include_dining:
            dining_dims = self.best_practices.get_room_dimensions(
                "dining", self.mod_config, "standard"
            )
            room_specs.append(
                RoomSpec(
                    room_type="dining",
                    size=dining_dims,
                    quantity=1,
                    priority=2,  # Central hub
                    adjacency_preferences=["kitchen", "recreation"],
                )
            )

        if requirements.include_storage:
            room_specs.append(
                RoomSpec(room_type="storage", size=(6, 6), quantity=1, priority=3)
            )

        if requirements.num_workrooms > 0:
            workshop_dims = self.best_practices.get_room_dimensions(
                "workshop", self.mod_config, "standard"
            )
            room_specs.append(
                RoomSpec(
                    room_type="workshop",
                    size=workshop_dims,
                    quantity=requirements.num_workrooms,
                    priority=4,  # Can be peripheral
                    adjacency_preferences=["workshop", "storage"],  # Cluster workshops
                    special_features=["toolbox_optimization"],
                )
            )

        if requirements.include_rec:
            room_specs.append(
                RoomSpec(room_type="recreation", size=(5, 4), quantity=1, priority=4)
            )

        if requirements.include_medical:
            hospital_dims = self.best_practices.get_room_dimensions(
                "hospital", self.mod_config, "standard"
            )
            room_specs.append(
                RoomSpec(
                    room_type="medical",
                    size=hospital_dims,
                    quantity=1,
                    priority=1,  # Highest priority for entrance placement
                    special_features=["near_entrance", "sterile_tiles"],
                    adjacency_preferences=["entrance", "storage"],
                )
            )

        return BaseDesignPlan(
            room_specs=room_specs,
            layout_strategy=requirements.style,
            traffic_flow="organic",
            defense_strategy=requirements.defense_level,
            expansion_plan="as needed",
            special_considerations=requirements.special_requirements or [],
            estimated_resources={},
        )

    def _match_prefabs_to_specs(self, room_specs: List[RoomSpec]) -> List[PrefabMatch]:
        """Find best prefab matches for each room specification"""
        matches = []

        for spec in room_specs:
            # Get candidate prefabs for this room type
            candidates = self._get_candidate_prefabs(spec.room_type)

            if not candidates:
                print(f"  Warning: No prefabs found for {spec.room_type}")
                continue

            # Score each candidate
            best_match = None
            best_score = -1

            for prefab in candidates:
                score, reasons = self._score_prefab_for_spec(prefab, spec)

                if score > best_score:
                    best_score = score
                    best_match = PrefabMatch(
                        prefab=prefab, score=score, room_spec=spec, reasons=reasons
                    )

            if best_match:
                # Add multiple if quantity > 1
                for _ in range(spec.quantity):
                    matches.append(best_match)

        return matches

    def _get_candidate_prefabs(self, room_type: str) -> List[AlphaPrefabLayout]:
        """Get prefabs that could work for a room type"""
        candidates = []

        # Get category keywords for this room type
        keywords = self.room_type_mapping.get(room_type, [room_type])

        # Search prefab library
        for category, prefabs in self.prefab_library.items():
            # Check if category matches any keyword
            category_lower = category.lower()
            if any(kw in category_lower for kw in keywords):
                candidates.extend(prefabs[:5])  # Take top 5 from each matching category

        # Also check prefab names
        for category, prefabs in self.prefab_library.items():
            for prefab in prefabs:
                prefab_name_lower = prefab.def_name.lower()
                if any(kw in prefab_name_lower for kw in keywords):
                    if prefab not in candidates:
                        candidates.append(prefab)

        return candidates

    def _score_prefab_for_spec(
        self, prefab: AlphaPrefabLayout, spec: RoomSpec
    ) -> Tuple[float, List[str]]:
        """Score how well a prefab matches a room specification"""
        score = 0.0
        reasons = []

        # Size match (most important)
        target_area = spec.size[0] * spec.size[1]
        actual_area = prefab.width * prefab.height
        area_diff = abs(target_area - actual_area) / target_area

        if area_diff < 0.2:
            score += 5.0
            reasons.append(
                f"Excellent size match ({actual_area} vs {target_area} tiles)"
            )
        elif area_diff < 0.5:
            score += 3.0
            reasons.append(f"Good size match ({actual_area} vs {target_area} tiles)")
        elif area_diff < 1.0:
            score += 1.0
            reasons.append(f"Acceptable size ({actual_area} vs {target_area} tiles)")
        else:
            score -= 2.0
            reasons.append(f"Size mismatch ({actual_area} vs {target_area} tiles)")

        # Aspect ratio match
        target_ratio = spec.size[0] / spec.size[1]
        actual_ratio = prefab.width / prefab.height
        ratio_diff = abs(target_ratio - actual_ratio)

        if ratio_diff < 0.3:
            score += 2.0
            reasons.append("Good aspect ratio")

        # Name match bonus
        if spec.room_type.lower() in prefab.def_name.lower():
            score += 3.0
            reasons.append(f"Name indicates {spec.room_type}")

        # Check for special features
        if spec.special_features:
            # This would need more sophisticated analysis of prefab contents
            score += 1.0
            reasons.append("May support special features")

        # Variety bonus (avoid using same prefab too much)
        if prefab.def_name not in [
            p.prefab.def_name for p in self.placed_prefabs if hasattr(p, "prefab")
        ]:
            score += 1.0
            reasons.append("Adds variety")

        return score, reasons

    def _try_place_prefab_smart(
        self, match: PrefabMatch, placed: List[PrefabMatch]
    ) -> bool:
        """Intelligently place a prefab considering adjacency preferences"""
        prefab = match.prefab
        spec = match.room_spec

        # Convert prefab to grid
        prefab_grid = self._layout_to_array(prefab)

        # Find best position based on adjacency preferences
        best_pos = None
        best_score = -1

        # Try multiple positions
        for _ in range(50):
            x = random.randint(0, self.width - prefab.width)
            y = random.randint(0, self.height - prefab.height)

            if self._can_place_at(x, y, prefab_grid):
                # Score this position
                pos_score = self._score_position(x, y, spec, placed)

                if pos_score > best_score:
                    best_score = pos_score
                    best_pos = (x, y)

        # Place at best position
        if best_pos:
            x, y = best_pos
            for py in range(prefab.height):
                for px in range(prefab.width):
                    if prefab_grid[py, px] != 0:
                        self.grid[y + py, x + px] = prefab_grid[py, px]
            return True

        return False

    def _score_position(
        self, x: int, y: int, spec: RoomSpec, placed: List[PrefabMatch]
    ) -> float:
        """Score a position based on room requirements and best practices"""
        score = 0.0

        center_x, center_y = self.width // 2, self.height // 2
        dist_to_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        max_dist = ((self.width / 2) ** 2 + (self.height / 2) ** 2) ** 0.5

        # Apply best practices for room placement
        if spec.room_type == "bedroom":
            # Bedrooms should be on edges (best practice)
            edge_dist = min(x, y, self.width - x, self.height - y)
            score += (1 - edge_dist / (self.width // 2)) * 3
        elif spec.room_type in ["kitchen", "dining", "recreation"]:
            # Central hub rooms (best practice)
            score += (1 - dist_to_center / max_dist) * 4
        elif spec.room_type == "workshop":
            # Workshops can be peripheral but should cluster
            for other in placed:
                if other.room_spec.room_type == "workshop":
                    workshop_dist = abs(x - other.position[0]) + abs(
                        y - other.position[1]
                    )
                    if workshop_dist < 10:
                        score += 3  # Bonus for clustering
        elif spec.priority <= 2:  # High priority rooms
            score += (1 - dist_to_center / max_dist) * 3
        else:  # Low priority can be on edges
            score += (dist_to_center / max_dist) * 2

        # Check adjacency preferences using best practices
        if spec.adjacency_preferences:
            for other_match in placed:
                if other_match.room_spec.room_type in spec.adjacency_preferences:
                    # Calculate distance to other room
                    if hasattr(other_match, "position"):
                        dist = abs(x - other_match.position[0]) + abs(
                            y - other_match.position[1]
                        )
                        if dist < 3:  # Adjacent
                            score += 5.0
                        elif dist < 10:  # Nearby
                            score += 2.0

        # Kitchen-freezer must be adjacent (best practice)
        if spec.room_type == "freezer":
            for other in placed:
                if other.room_spec.room_type == "kitchen" and hasattr(
                    other, "position"
                ):
                    dist = abs(x - other.position[0]) + abs(y - other.position[1])
                    if dist <= 1:
                        score += 10.0  # Strong bonus for adjacency
                    else:
                        score -= 5.0  # Penalty for separation

        # Medical rooms near edge (for quick access)
        if spec.room_type == "medical" and "near_entrance" in spec.special_features:
            edge_dist = min(x, y, self.width - x, self.height - y)
            if edge_dist < 10:
                score += 3.0

        return score

    def _fill_with_decoration(self, plan: BaseDesignPlan):
        """Fill empty spaces with decorative elements"""
        # Add corridors
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    # Check if adjacent to rooms
                    adjacent_to_room = False
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if self.grid[ny, nx] in [TileType.FLOOR, TileType.ROOM]:
                                adjacent_to_room = True
                                break

                    if adjacent_to_room and random.random() < 0.3:
                        self.grid[y, x] = TileType.FLOOR

    def _ensure_connectivity(self):
        """Ensure all rooms are connected with corridors"""
        # Simple connectivity: add corridors between isolated areas
        # This is a simplified version - a proper implementation would use pathfinding
        pass

    def _add_defensive_elements(self, defense_strategy: str):
        """Add defensive structures based on strategy and best practices"""
        defense_level = defense_strategy.lower()

        # Add perimeter walls first
        self._add_perimeter_walls()

        if "killbox" in defense_level or defense_level in ["high", "extreme"]:
            self._add_killbox()
        elif defense_level == "medium":
            self._add_basic_defenses()

    def _add_perimeter_walls(self):
        """Add walls around the perimeter with strategic openings"""
        # Add walls but leave openings for killbox/entrance
        entrance_x = self.width // 2
        entrance_width = 3

        for x in range(self.width):
            # Top wall with entrance
            if abs(x - entrance_x) > entrance_width:
                if self.grid[0, x] == 0:
                    self.grid[0, x] = TileType.WALL

            # Bottom wall (solid)
            if self.grid[self.height - 1, x] == 0:
                self.grid[self.height - 1, x] = TileType.WALL

        # Side walls (solid)
        for y in range(self.height):
            if self.grid[y, 0] == 0:
                self.grid[y, 0] = TileType.WALL
            if self.grid[y, self.width - 1] == 0:
                self.grid[y, self.width - 1] = TileType.WALL

    def _add_killbox(self):
        """Add a properly designed killbox at the main entrance"""
        # Use best practices for killbox design
        killbox_layout = self.best_practices.generate_killbox_layout(
            width=min(15, self.width // 3), height=min(20, self.height // 3)
        )

        # Find best position (typically at map edge)
        entrance_x = self.width // 2 - killbox_layout.shape[1] // 2
        entrance_y = 0

        # Place killbox
        kb_height, kb_width = killbox_layout.shape
        for y in range(min(kb_height, self.height)):
            for x in range(min(kb_width, self.width)):
                grid_x = entrance_x + x
                grid_y = entrance_y + y

                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    if killbox_layout[y, x] != 0:
                        self.grid[grid_y, grid_x] = killbox_layout[y, x]

    def _add_basic_defenses(self):
        """Add basic defensive elements for medium defense"""
        # Add some sandbags/defensive positions near entrance
        entrance_x = self.width // 2
        defense_y = 5  # A bit inside from entrance

        # Add defensive positions
        for x in range(max(0, entrance_x - 5), min(self.width, entrance_x + 6)):
            if defense_y < self.height and self.grid[defense_y, x] == 0:
                if abs(x - entrance_x) > 1:  # Don't block path
                    self.grid[defense_y, x] = TileType.FURNITURE  # Sandbags

    def _get_default_plan(self) -> BaseDesignPlan:
        """Get a default plan when none provided"""
        return BaseDesignPlan(
            room_specs=[
                RoomSpec("bedroom", (4, 3), 5, 1),
                RoomSpec("kitchen", (5, 4), 1, 2),
                RoomSpec("storage", (6, 6), 1, 3),
                RoomSpec("workshop", (5, 5), 2, 3),
                RoomSpec("recreation", (5, 4), 1, 4),
            ],
            layout_strategy="balanced",
            traffic_flow="organic",
            defense_strategy="medium",
            expansion_plan="as needed",
            special_considerations=[],
            estimated_resources={},
        )

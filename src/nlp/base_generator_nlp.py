"""
Natural Language Processing interface for base generation.
Allows users to describe their desired base in natural language.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from src.generators.improved_wfc_generator import ImprovedWFCGenerator
from src.parser.save_parser import RimWorldSaveParser as SaveParser
from src.models.game_entities import BuildingType


@dataclass
class BaseRequirements:
    """Parsed requirements from natural language input"""

    num_colonists: int = 5
    num_bedrooms: int = 5
    num_workrooms: int = 2
    include_kitchen: bool = True
    include_dining: bool = True
    include_rec: bool = True
    include_storage: bool = True
    include_medical: bool = True
    include_research: bool = False
    include_prison: bool = False
    defense_level: str = "medium"  # low, medium, high
    style: str = "efficient"  # efficient, spacious, compact, defensive
    special_requirements: list[str] = field(default_factory=list)
    # Zone-based requirements
    use_outer_areas: bool = False
    agriculture_zones: bool = False
    defensive_layers: int = 1  # 1, 2, or 3 layers
    bridge_water: bool = False  # Build bridges to expand over water
    zone_layout: str = "single"  # single, multi-zone, concentric, scattered


class BaseGeneratorNLP:
    """Natural language interface for base generation"""

    # Keywords for parsing
    ROOM_KEYWORDS = {
        "bedroom": ["bedroom", "bed", "sleeping", "quarters", "barracks"],
        "kitchen": ["kitchen", "cooking", "stove", "food prep"],
        "dining": ["dining", "eating", "cafeteria", "mess hall"],
        "recreation": ["recreation", "rec", "entertainment", "fun", "joy", "games"],
        "storage": ["storage", "warehouse", "stockpile", "inventory"],
        "workshop": [
            "workshop",
            "production",
            "crafting",
            "manufacturing",
            "workbench",
        ],
        "medical": ["medical", "hospital", "infirmary", "medbay", "health"],
        "research": ["research", "lab", "laboratory", "science"],
        "prison": ["prison", "jail", "detention", "prisoner"],
        "power": ["power", "generator", "battery", "electricity"],
    }

    STYLE_KEYWORDS = {
        "efficient": ["efficient", "optimized", "practical", "functional"],
        "spacious": ["spacious", "roomy", "large", "big", "comfortable"],
        "compact": ["compact", "small", "tight", "minimal", "tiny"],
        "defensive": ["defensive", "fortified", "secure", "protected", "safe"],
    }

    DEFENSE_KEYWORDS = {
        "high": [
            "heavily defended",
            "maximum defense",
            "fortress",
            "highly secure",
            "impenetrable",
        ],
        "medium": ["defended", "secure", "protected", "some defense"],
        "low": ["minimal defense", "open", "peaceful", "no defense"],
    }

    ZONE_KEYWORDS = {
        "use_outer": [
            "outer area",
            "outer zone",
            "expand outward",
            "use all land",
            "entire map",
            "all buildable",
            "whole atoll",
            "outer edge",
            "perimeter",
        ],
        "agriculture": [
            "farm",
            "agriculture",
            "growing",
            "crops",
            "greenhouse",
            "food production",
            "farmland",
            "cultivate",
            "harvest",
        ],
        "multi_defense": [
            "multiple walls",
            "layered defense",
            "concentric",
            "outer wall",
            "inner defense",
            "defensive layers",
            "multiple perimeters",
        ],
        "bridge": [
            "bridge",
            "over water",
            "shallow water",
            "expand onto water",
            "build on water",
            "connect islands",
            "water building",
        ],
        "multi_zone": [
            "multiple zones",
            "separate areas",
            "distributed",
            "spread out",
            "zones",
            "district",
            "sectors",
            "divided base",
        ],
    }

    def __init__(self, save_path: str | None = None):
        """
        Initialize NLP interface.

        Args:
            save_path: Optional path to save file for context
        """
        self.save_path = save_path
        self.parser = SaveParser() if save_path else None
        self.parse_result = None

        if self.save_path and Path(self.save_path).exists() and self.parser:
            self.parse_result = self.parser.parse(self.save_path)

    def parse_request(self, user_input: str) -> BaseRequirements:
        """
        Parse natural language request into base requirements.

        Args:
            user_input: Natural language description of desired base

        Returns:
            Parsed BaseRequirements
        """
        requirements = BaseRequirements()
        input_lower = user_input.lower()

        # Parse number of colonists
        colonist_match = re.search(r"(\d+)\s*(?:colonist|people|pawn)", input_lower)
        if colonist_match:
            requirements.num_colonists = int(colonist_match.group(1))
            requirements.num_bedrooms = requirements.num_colonists

        # Parse number of specific rooms
        bedroom_match = re.search(r"(\d+)\s*bedroom", input_lower)
        if bedroom_match:
            requirements.num_bedrooms = int(bedroom_match.group(1))

        workshop_match = re.search(
            r"(\d+)\s*(?:workshop|workroom|production)", input_lower
        )
        if workshop_match:
            requirements.num_workrooms = int(workshop_match.group(1))

        # Parse room inclusions
        requirements.include_kitchen = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["kitchen"]
        )
        requirements.include_dining = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["dining"]
        )
        requirements.include_rec = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["recreation"]
        )
        requirements.include_storage = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["storage"]
        )
        requirements.include_medical = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["medical"]
        )
        requirements.include_research = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["research"]
        )
        requirements.include_prison = self._check_keywords(
            input_lower, self.ROOM_KEYWORDS["prison"]
        )

        # Parse style
        for style, keywords in self.STYLE_KEYWORDS.items():
            if self._check_keywords(input_lower, keywords):
                requirements.style = style
                break

        # Parse defense level
        for level, keywords in self.DEFENSE_KEYWORDS.items():
            if self._check_keywords(input_lower, keywords):
                requirements.defense_level = level
                break

        # Parse zone-based requirements
        requirements.use_outer_areas = self._check_keywords(
            input_lower, self.ZONE_KEYWORDS["use_outer"]
        )
        requirements.agriculture_zones = self._check_keywords(
            input_lower, self.ZONE_KEYWORDS["agriculture"]
        )
        requirements.bridge_water = self._check_keywords(
            input_lower, self.ZONE_KEYWORDS["bridge"]
        )

        # Parse defensive layers
        if self._check_keywords(input_lower, self.ZONE_KEYWORDS["multi_defense"]):
            requirements.defensive_layers = 2
            if "three" in input_lower or "triple" in input_lower:
                requirements.defensive_layers = 3

        # Parse zone layout
        if self._check_keywords(input_lower, self.ZONE_KEYWORDS["multi_zone"]):
            requirements.zone_layout = "multi-zone"
        elif "concentric" in input_lower:
            requirements.zone_layout = "concentric"
        elif "scattered" in input_lower or "distributed" in input_lower:
            requirements.zone_layout = "scattered"

        # Extract special requirements
        requirements.special_requirements = self._extract_special_requirements(
            input_lower
        )

        # Apply style presets
        self._apply_style_presets(requirements)

        return requirements

    def _check_keywords(self, text: str, keywords: list[str]) -> bool:
        """Check if any keyword appears in text"""
        return any(keyword in text for keyword in keywords)

    def _extract_special_requirements(self, text: str) -> list[str]:
        """Extract any special requirements from text"""
        special = []

        if "killbox" in text:
            special.append("killbox")
        if "freezer" in text:
            special.append("freezer")
        if "greenhouse" in text:
            special.append("greenhouse")
        if "throne" in text or "royal" in text:
            special.append("throne_room")
        if "separate" in text and "kitchen" in text:
            special.append("separate_kitchen_freezer")
        if "central" in text and "courtyard" in text:
            special.append("central_courtyard")

        return special

    def _apply_style_presets(self, requirements: BaseRequirements):
        """Apply style-based presets to requirements"""
        if requirements.style == "defensive":
            requirements.defense_level = "high"
            if "killbox" not in (requirements.special_requirements or []):
                if requirements.special_requirements is None:
                    requirements.special_requirements = []
                requirements.special_requirements.append("killbox")

        elif requirements.style == "spacious":
            # Increase room counts for spacious style
            requirements.num_bedrooms = max(
                requirements.num_bedrooms, requirements.num_colonists + 2
            )
            requirements.include_rec = True
            requirements.include_dining = True

        elif requirements.style == "compact":
            # Reduce rooms for compact style
            requirements.num_bedrooms = min(
                requirements.num_bedrooms, requirements.num_colonists
            )
            requirements.num_workrooms = max(1, requirements.num_workrooms - 1)

    def generate_base(
        self, user_input: str, width: int = 50, height: int = 50
    ) -> tuple[np.ndarray, str]:
        """
        Generate a base from natural language description.

        Args:
            user_input: Natural language description
            width: Base width
            height: Base height

        Returns:
            Tuple of (generated grid, description of what was created)
        """
        # Parse requirements
        requirements = self.parse_request(user_input)

        # Load learned patterns if available
        patterns_file = Path("learned_patterns_alpha.json")
        if not patterns_file.exists():
            patterns_file = Path("learned_patterns.json")

        # Create generator
        generator = ImprovedWFCGenerator(width, height, patterns_file)

        # Generate buildable mask if we have save data
        buildable_mask = None
        if self.parse_result and self.parse_result.maps:
            # Get bridge positions from first map
            first_map = self.parse_result.maps[0]
            bridge_positions = set()
            for building in first_map.buildings:
                if building.building_type == BuildingType.BRIDGE:
                    bridge_positions.add((building.position.x, building.position.y))

            if bridge_positions:
                # Create buildable mask
                buildable_mask = np.zeros((height, width), dtype=bool)
                for x, y in bridge_positions:
                    if 0 <= x < width and 0 <= y < height:
                        buildable_mask[y, x] = True

        # Generate base
        grid = generator.generate_with_templates(
            buildable_mask=buildable_mask,
            num_bedrooms=requirements.num_bedrooms,
            num_workrooms=requirements.num_workrooms,
            include_kitchen=requirements.include_kitchen,
            include_rec=requirements.include_rec,
            include_storage=requirements.include_storage,
        )

        # Generate description
        description = self._generate_description(requirements, generator)

        return grid, description

    def _generate_description(
        self, requirements: BaseRequirements, generator: ImprovedWFCGenerator
    ) -> str:
        """Generate description of what was created"""
        lines = []
        lines.append(
            f"Generated {requirements.style} base for {requirements.num_colonists} colonists:"
        )
        lines.append(f"- {requirements.num_bedrooms} bedrooms")

        if requirements.include_kitchen:
            lines.append("- Kitchen and dining area")
        if requirements.include_storage:
            lines.append("- Storage facilities")
        if requirements.num_workrooms > 0:
            lines.append(f"- {requirements.num_workrooms} workshop(s)")
        if requirements.include_rec:
            lines.append("- Recreation room")
        if requirements.include_medical:
            lines.append("- Medical bay")
        if requirements.include_research:
            lines.append("- Research laboratory")
        if requirements.include_prison:
            lines.append("- Prison cells")

        lines.append(f"- Defense level: {requirements.defense_level}")

        if requirements.special_requirements:
            lines.append("Special features:")
            for req in requirements.special_requirements:
                lines.append(f"  - {req.replace('_', ' ').title()}")

        # Add room placement info
        if generator.room_placements:
            lines.append(
                f"\nPlaced {len(generator.room_placements)} rooms successfully"
            )

        return "\n".join(lines)

    def interpret_feedback(
        self, feedback: str, current_requirements: BaseRequirements
    ) -> BaseRequirements:
        """
        Interpret user feedback and adjust requirements.

        Args:
            feedback: User feedback on generated base
            current_requirements: Current requirements

        Returns:
            Updated requirements
        """
        feedback_lower = feedback.lower()
        updated = BaseRequirements(**current_requirements.__dict__)

        # Check for size adjustments
        if "bigger" in feedback_lower or "larger" in feedback_lower:
            updated.style = "spacious"
            updated.num_bedrooms += 2
        elif "smaller" in feedback_lower or "compact" in feedback_lower:
            updated.style = "compact"
            updated.num_bedrooms = max(1, updated.num_bedrooms - 1)

        # Check for room additions
        if (
            "add" in feedback_lower
            or "need" in feedback_lower
            or "want" in feedback_lower
        ):
            for room_type, keywords in self.ROOM_KEYWORDS.items():
                if self._check_keywords(feedback_lower, keywords):
                    setattr(updated, f"include_{room_type}", True)

        # Check for room removals
        if (
            "remove" in feedback_lower
            or "don't need" in feedback_lower
            or "no" in feedback_lower
        ):
            for room_type, keywords in self.ROOM_KEYWORDS.items():
                if self._check_keywords(feedback_lower, keywords):
                    setattr(updated, f"include_{room_type}", False)

        # Check for defense adjustments
        if "more defense" in feedback_lower or "better defense" in feedback_lower:
            updated.defense_level = "high"
        elif "less defense" in feedback_lower:
            updated.defense_level = "low"

        return updated


class NLPExamples:
    """Example natural language inputs for testing"""

    EXAMPLES = [
        "Create an efficient base for 8 colonists with good defenses",
        "I need a compact base for 5 people with a kitchen, storage, and 2 workshops",
        "Design a spacious fortress for 12 colonists with maximum defense and a killbox",
        "Make a small peaceful base for 3 colonists with minimal defense",
        "Build a base with 6 bedrooms, medical bay, research lab, and recreation room",
        "Create a defensive base with separate kitchen and freezer, central courtyard",
        "I want a base for 10 colonists with prison cells and throne room",
        "Design an optimized production base with 4 workshops and large storage",
        "Make a cozy base for 4 people with dining room and recreation area",
        "Create a self-sufficient base with greenhouse, kitchen, and 5 bedrooms",
    ]

    @staticmethod
    def test_parsing():
        """Test parsing of example inputs"""
        nlp = BaseGeneratorNLP()

        print("Testing NLP parsing:\n")
        for example in NLPExamples.EXAMPLES:
            print(f"Input: {example}")
            requirements = nlp.parse_request(example)
            print(f"Parsed: {requirements}\n")


if __name__ == "__main__":
    # Test the NLP interface
    NLPExamples.test_parsing()

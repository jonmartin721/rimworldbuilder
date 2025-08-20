"""
Claude API integration for intelligent RimWorld base design.
Uses Claude to generate detailed base plans based on requirements.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: anthropic package not installed. Using fallback designs.")

try:
    from src.utils.progress import spinner
except ImportError:
    # Fallback if progress utils not available
    from contextlib import contextmanager
    @contextmanager
    def spinner(message):
        print(f"{message}...")
        yield


@dataclass
class BaseDesignRequest:
    """Request for Claude to design a base"""
    colonist_count: int
    map_size: Tuple[int, int]
    biome: str = "temperate_forest"
    difficulty: str = "medium"
    priorities: List[str] = None  # e.g., ["defense", "efficiency", "aesthetics"]
    special_requirements: List[str] = None  # e.g., ["killbox", "throne room", "hospital"]
    available_space: Optional[Tuple[int, int]] = None  # Buildable area dimensions
    existing_structures: Optional[str] = None  # Description of what's already built


@dataclass
class RoomSpec:
    """Specification for a room"""
    room_type: str
    size: Tuple[int, int]  # (width, height)
    quantity: int
    priority: int  # 1 = highest priority
    adjacency_preferences: List[str] = None
    special_features: List[str] = None


@dataclass
class BaseDesignPlan:
    """Complete base design plan from Claude"""
    room_specs: List[RoomSpec]
    layout_strategy: str  # e.g., "centralized", "distributed", "defensive_layers"
    traffic_flow: str  # Description of main pathways
    defense_strategy: str
    expansion_plan: str
    special_considerations: List[str]
    estimated_resources: Dict[str, int]


class ClaudeBaseDesigner:
    """Uses Claude API to generate intelligent base designs"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude designer.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            print("Warning: No API key provided. Set ANTHROPIC_API_KEY or pass api_key parameter.")
    
    def design_base(self, request: BaseDesignRequest) -> Optional[BaseDesignPlan]:
        """
        Get Claude to design a base based on requirements.
        
        Args:
            request: Base design requirements
            
        Returns:
            Complete base design plan or None if API not available
        """
        if not self.client:
            return self._get_fallback_design(request)
        
        prompt = self._create_design_prompt(request)
        
        try:
            with spinner("Requesting base design from Claude AI"):
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Using Haiku for cost efficiency
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                time.sleep(0.5)  # Give spinner time to show
            
            print("✅ Received AI design plan")
            
            # Parse Claude's response
            return self._parse_claude_response(response.content[0].text)
            
        except Exception as e:
            print(f"⚠️ Claude API error: {e}")
            print("Using fallback design...")
            return self._get_fallback_design(request)
    
    def _create_design_prompt(self, request: BaseDesignRequest) -> str:
        """Create detailed prompt for Claude"""
        prompt = f"""You are an expert RimWorld base designer. Design an optimal base layout for the following requirements:

REQUIREMENTS:
- Colonists: {request.colonist_count}
- Map size: {request.map_size[0]}x{request.map_size[1]} tiles
- Biome: {request.biome}
- Difficulty: {request.difficulty}
- Priorities: {', '.join(request.priorities or ['balanced'])}
- Special requirements: {', '.join(request.special_requirements or ['none'])}
"""
        
        if request.available_space:
            prompt += f"- Buildable area: {request.available_space[0]}x{request.available_space[1]} tiles\n"
        
        if request.existing_structures:
            prompt += f"- Existing structures: {request.existing_structures}\n"
        
        prompt += """
Please provide a detailed base design in the following JSON format:
{
    "room_specs": [
        {
            "room_type": "bedroom",
            "size": [4, 3],
            "quantity": 5,
            "priority": 1,
            "adjacency_preferences": ["corridor", "dining"],
            "special_features": ["double_bed", "dresser"]
        }
    ],
    "layout_strategy": "defensive_onion_layers",
    "traffic_flow": "Central corridor with branching wings",
    "defense_strategy": "Killbox at main entrance with fallback positions",
    "expansion_plan": "Reserve northern area for future workshop expansion",
    "special_considerations": ["Place hospital near entrance for quick rescue", "Kitchen adjacent to freezer"],
    "estimated_resources": {"steel": 500, "wood": 800, "stone_blocks": 400}
}

Consider RimWorld best practices:
- Bedrooms: 4x3 minimum for mood bonus
- Kitchen near freezer and dining room
- Hospital accessible from entrance
- Workshops clustered for efficiency
- Recreation centrally located
- Prison separate from colonist areas
- Power generation protected but accessible
- Storage distributed based on item types

Provide only the JSON response, no additional text."""
        
        return prompt
    
    def _parse_claude_response(self, response_text: str) -> BaseDesignPlan:
        """Parse Claude's JSON response into BaseDesignPlan"""
        try:
            # Extract JSON from response (Claude might add text around it)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Convert to RoomSpec objects
                room_specs = []
                for spec in data.get('room_specs', []):
                    room_specs.append(RoomSpec(
                        room_type=spec['room_type'],
                        size=tuple(spec['size']),
                        quantity=spec['quantity'],
                        priority=spec['priority'],
                        adjacency_preferences=spec.get('adjacency_preferences', []),
                        special_features=spec.get('special_features', [])
                    ))
                
                return BaseDesignPlan(
                    room_specs=room_specs,
                    layout_strategy=data.get('layout_strategy', 'balanced'),
                    traffic_flow=data.get('traffic_flow', 'organic'),
                    defense_strategy=data.get('defense_strategy', 'basic walls'),
                    expansion_plan=data.get('expansion_plan', 'as needed'),
                    special_considerations=data.get('special_considerations', []),
                    estimated_resources=data.get('estimated_resources', {})
                )
        except Exception as e:
            print(f"Error parsing Claude response: {e}")
            return self._get_fallback_design(BaseDesignRequest(colonist_count=5))
    
    def _get_fallback_design(self, request: BaseDesignRequest) -> BaseDesignPlan:
        """Fallback design when API is not available"""
        # Calculate room needs based on colonist count
        bedroom_count = request.colonist_count
        
        room_specs = [
            RoomSpec("bedroom", (4, 3), bedroom_count, 1, ["corridor"]),
            RoomSpec("kitchen", (6, 5), 1, 2, ["dining", "freezer"]),
            RoomSpec("dining", (7, 5), 1, 2, ["kitchen", "recreation"]),
            RoomSpec("storage", (8, 6), 2, 3, ["workshop"]),
            RoomSpec("workshop", (6, 6), 2, 3, ["storage"]),
            RoomSpec("hospital", (5, 4), 1, 2, ["entrance"]),
            RoomSpec("recreation", (6, 5), 1, 4, ["dining", "bedroom"]),
        ]
        
        # Add based on priorities
        if request.priorities and "defense" in request.priorities:
            room_specs.append(RoomSpec("killbox", (10, 8), 1, 1, ["entrance"]))
            room_specs.append(RoomSpec("armory", (4, 4), 1, 2, ["killbox"]))
        
        if request.special_requirements:
            if "throne_room" in request.special_requirements:
                room_specs.append(RoomSpec("throne_room", (8, 8), 1, 2, ["entrance"]))
            if "prison" in request.special_requirements:
                room_specs.append(RoomSpec("prison", (5, 4), 2, 3, ["entrance"]))
        
        return BaseDesignPlan(
            room_specs=room_specs,
            layout_strategy="centralized" if bedroom_count <= 8 else "distributed",
            traffic_flow="Central corridor with room wings",
            defense_strategy="Perimeter wall with single entrance",
            expansion_plan="Reserve edges for future expansion",
            special_considerations=[
                "Place bedrooms away from workshop noise",
                "Kitchen and freezer must be adjacent",
                "Hospital near entrance for emergency access"
            ],
            estimated_resources={
                "steel": 100 * bedroom_count,
                "wood": 150 * bedroom_count,
                "stone_blocks": 200 * bedroom_count
            }
        )
    
    def refine_design(self, plan: BaseDesignPlan, feedback: str) -> BaseDesignPlan:
        """
        Refine a design based on user feedback.
        
        Args:
            plan: Current design plan
            feedback: User feedback on what to change
            
        Returns:
            Refined design plan
        """
        if not self.client:
            # Simple rule-based refinement without API
            return self._simple_refinement(plan, feedback)
        
        prompt = f"""Current RimWorld base design:
{json.dumps(asdict(plan), indent=2)}

User feedback: {feedback}

Please refine the design based on this feedback. Maintain the same JSON format.
Focus on addressing the specific concerns while keeping what works well.
Provide only the JSON response."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_claude_response(response.content[0].text)
            
        except Exception as e:
            print(f"Claude API error: {e}")
            return self._simple_refinement(plan, feedback)
    
    def _simple_refinement(self, plan: BaseDesignPlan, feedback: str) -> BaseDesignPlan:
        """Simple rule-based refinement without API"""
        feedback_lower = feedback.lower()
        
        # Copy current plan
        new_specs = list(plan.room_specs)
        
        # Check for size adjustments
        if "bigger" in feedback_lower or "larger" in feedback_lower:
            for spec in new_specs:
                spec.size = (spec.size[0] + 1, spec.size[1] + 1)
        elif "smaller" in feedback_lower or "compact" in feedback_lower:
            for spec in new_specs:
                spec.size = (max(3, spec.size[0] - 1), max(3, spec.size[1] - 1))
        
        # Check for room additions
        if "more storage" in feedback_lower:
            storage_specs = [s for s in new_specs if s.room_type == "storage"]
            if storage_specs:
                storage_specs[0].quantity += 1
        
        if "more bedrooms" in feedback_lower:
            bedroom_specs = [s for s in new_specs if s.room_type == "bedroom"]
            if bedroom_specs:
                bedroom_specs[0].quantity += 2
        
        # Check for defense improvements
        if "more defense" in feedback_lower or "better defense" in feedback_lower:
            plan.defense_strategy = "Multiple defensive layers with killbox and turrets"
            # Add defensive rooms if not present
            if not any(s.room_type == "killbox" for s in new_specs):
                new_specs.append(RoomSpec("killbox", (10, 8), 1, 1, ["entrance"]))
        
        plan.room_specs = new_specs
        return plan


def demo_claude_designer():
    """Demonstrate the Claude base designer"""
    designer = ClaudeBaseDesigner()
    
    # Create a request
    request = BaseDesignRequest(
        colonist_count=8,
        map_size=(250, 250),
        biome="temperate_forest",
        difficulty="hard",
        priorities=["defense", "efficiency"],
        special_requirements=["killbox", "hospital", "prison"],
        available_space=(50, 50)
    )
    
    print("Requesting base design from Claude...")
    plan = designer.design_base(request)
    
    if plan:
        print("\n=== BASE DESIGN PLAN ===")
        print(f"Layout Strategy: {plan.layout_strategy}")
        print(f"Defense Strategy: {plan.defense_strategy}")
        print(f"Traffic Flow: {plan.traffic_flow}")
        print("\nRoom Specifications:")
        for spec in plan.room_specs:
            print(f"  - {spec.quantity}x {spec.room_type} ({spec.size[0]}x{spec.size[1]})")
        print(f"\nEstimated Resources: {plan.estimated_resources}")
        
        # Test refinement
        print("\n=== REFINING DESIGN ===")
        refined = designer.refine_design(plan, "Make bedrooms bigger and add more storage")
        print("Refined room specs:")
        for spec in refined.room_specs:
            print(f"  - {spec.quantity}x {spec.room_type} ({spec.size[0]}x{spec.size[1]})")


if __name__ == "__main__":
    demo_claude_designer()
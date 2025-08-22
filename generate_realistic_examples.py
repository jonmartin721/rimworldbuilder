#!/usr/bin/env python3
"""Generate realistic example bases using NLP requests"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.generators.realistic_base_generator import RealisticBaseGenerator
from src.visualization.realistic_visualizer import RealisticBaseVisualizer


def parse_nlp_request(request: str) -> dict:
    """Simple NLP parser to extract base requirements"""
    request_lower = request.lower()
    
    # Extract number of colonists/bedrooms
    import re
    colonist_match = re.search(r'(\d+)\s*colonist', request_lower)
    num_bedrooms = int(colonist_match.group(1)) if colonist_match else 5
    
    # Check for keywords
    params = {
        "num_bedrooms": num_bedrooms,
        "include_kitchen": "kitchen" in request_lower or "food" in request_lower or True,
        "include_dining": "dining" in request_lower or "eating" in request_lower,
        "include_workshop": 4 if "production" in request_lower else 2,
        "include_storage": "storage" in request_lower or "warehouse" in request_lower or True,
        "include_hospital": "medical" in request_lower or "hospital" in request_lower
    }
    
    # Adjust based on style
    if "compact" in request_lower:
        params["num_bedrooms"] = min(params["num_bedrooms"], 4)
        params["include_workshop"] = 1
    elif "spacious" in request_lower or "large" in request_lower:
        params["num_bedrooms"] += 2
        params["include_dining"] = True
    
    if "defensive" in request_lower or "killbox" in request_lower:
        params["include_storage"] = True  # For weapons
        
    if "agricultural" in request_lower or "farm" in request_lower:
        params["include_storage"] = True  # For crops
        
    return params


def main():
    """Generate various realistic example bases from NLP requests"""
    
    # Create visualizer
    visualizer = RealisticBaseVisualizer(scale=10)
    
    # NLP requests similar to the original
    examples = [
        ("Compact defensive base for 6 colonists with killbox", 80, 80, "output/nlp_defensive.png"),
        ("Spacious production base with 4 workshops and large storage", 100, 100, "output/nlp_production.png"),
        ("Agricultural base for 8 colonists with storage and kitchen", 90, 90, "output/nlp_agricultural.png"),
        ("Efficient base for 10 colonists with medical bay and research", 100, 100, "output/nlp_medical.png"),
        ("Comfortable base with recreation room for 7 colonists", 85, 85, "output/nlp_comfort.png"),
        ("Small starter base for 3 colonists", 60, 60, "output/nlp_starter.png"),
        ("Large fortress for 12 colonists with hospital", 120, 120, "output/nlp_fortress.png"),
        ("Minimalist base for 4 colonists", 70, 70, "output/nlp_minimal.png"),
    ]
    
    print("=" * 60)
    print("REALISTIC NLP-BASED RIMWORLD BASE GENERATOR")
    print("=" * 60)
    
    for i, (request, width, height, filename) in enumerate(examples, 1):
        print(f"\n{i}. Request: '{request[:50]}...'")
        print("-" * 50)
        
        # Parse NLP request
        params = parse_nlp_request(request)
        print("Parsed parameters:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Generate base
        generator = RealisticBaseGenerator(width, height)
        grid = generator.generate_base(**params)
        
        # Save visualization
        visualizer.visualize(grid, filename, title=request[:40])
        
        # Print summary
        print(f"Generated base with {len(generator.rooms)} rooms")
        print(f"[OK] Saved to {filename}")
        
        # Show small ASCII preview
        print("ASCII Preview (center 30x15):")
        center_y = height // 2 - 7
        center_x = width // 2 - 15
        preview_grid = grid[center_y:center_y+15, center_x:center_x+30]
        preview = visualizer.visualize_ascii(preview_grid)
        for line in preview.split('\n'):
            print(f"  {line}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All NLP examples generated!")
    print("Check output/ folder for PNG files")
    print("=" * 60)


if __name__ == "__main__":
    main()
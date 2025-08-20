# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RimWorld Base Assistant - An AI tool for parsing RimWorld save files and generating optimized base layouts using procedural generation. Currently parses 3,400+ buildings including critical Frame_HeavyBridge structures for buildable areas.

## Essential Commands

### Development & Testing
```bash
# Install dependencies (uses Poetry)
poetry install

# Parse a save file (place .rws file in data/saves/)
poetry run python scripts/test_parser.py

# Generate visualizations
poetry run python scripts/layered_visualize.py  # Creates separate PNG for each building layer
poetry run python scripts/full_map_visualize.py  # Full map + detailed view

# Explore structure types in save file
poetry run python scripts/explore_structures.py
```

### Running Scripts
All scripts require the project root in Python path. Scripts in `scripts/` folder include:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Architecture & Key Implementation Details

### Save File Parser (`src/parser/save_parser.py`)
- **Critical:** Frame_HeavyBridge objects use `Class="Frame"` not `Class="Building"` - parser must check both
- Building type detection uses both exact and partial matching in `building_type_mapping`
- Position format in saves: `(x, 0, z)` where z is the Y coordinate on the map (not `(x, y, z)`)
- Terrain grid is base64 + zlib compressed (currently not fully decoded)

### Data Models (`src/models/game_entities.py`)
- `BuildingType` enum includes BRIDGE, FLOOR, CONDUIT, FENCE, LIGHT beyond basic types
- Position model uses x,y for map coordinates, z for elevation
- Buildings have `is_blueprint` and `is_frame` flags for construction status

### Visualization System
- **Layered approach:** Each building type can be rendered as separate layer
- Bridge detection is critical - shows buildable areas over water/terrain
- Power conduits (1000+) and walls (900+) dominate the visual space
- PNG visualizations work better than ASCII for complex bases

### Current Parse Statistics (from test save)
- 73 Frame_HeavyBridge (buildable area - critical for base planning)
- 1033 PowerConduits
- 972 Walls
- 176 active mods detected
- Map size: 250x250 tiles

## Known Issues & Important Notes

1. **Terrain decompression fails** with zlib header error - grid data needs different decompression approach
2. **ASCII visualizations** don't render well for complex bases - use PNG visualizations instead
3. **Colonist detection** checks faction starting with "Faction_" and def containing "Human"
4. **Memory usage:** Large saves (100MB+) load entirely into memory - streaming parser exists but not used

## Development Phases (from docs/DEV_PLAN.md)

- Phase 1: ✅ Parser & Models
- Phase 2: 🚧 Mod System Integration
- Phase 3: 📅 Wave Function Collapse Generator (needs bridge positions as buildable areas)
- Phase 4: 📅 Natural Language Interface

## File Organization

- Save files: `data/saves/*.rws` (gitignored)
- Visualizations: Generated as `base_*.png` (gitignored)
- Scripts: All in `scripts/` with path adjustment for imports
- Core logic: `src/parser/` and `src/models/`
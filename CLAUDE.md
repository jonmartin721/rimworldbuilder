# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RimWorld Base Assistant - A comprehensive AI-powered tool for parsing RimWorld save files and generating optimized base layouts using real RimWorld prefabs from the AlphaPrefabs mod. The system uses natural language processing and Claude AI to understand requirements, then intelligently selects and places appropriate prefabs to create functional bases that match user specifications.

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

### Main Interfaces
```bash
# Run the command-line interface
poetry run python rimworld_assistant.py

# Run the GUI interface (tkinter-based)
poetry run python rimworld_assistant_gui.py
```

### Key Feature: Smart Generation Pipeline
The system now features an intelligent generation pipeline:
1. **Natural Language Input** → User describes base in plain English
2. **NLP Parsing** → Extracts requirements (colonists, rooms, defense level)
3. **Claude AI Planning** (optional) → Creates detailed room specifications
4. **Prefab Matching** → Scores and selects best prefabs for each requirement
5. **Smart Placement** → Places prefabs considering adjacency and priorities
6. **Decorative Fill** → Adds corridors and details to complete the base

Use menu option 7 "Smart Generate (NLP → Prefabs)" for this feature.

### Running Scripts
Scripts in `scripts/` folder need path adjustment (now fixed in test_parser.py):
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

## User Experience Features

### Progress Indicators
The system now provides comprehensive feedback during all operations:
- **Animated Spinners** (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) for long operations
- **Progress Bars** with percentage for multi-step processes
- **Step-by-step logging** showing current operation
- **Time tracking** for performance monitoring
- **Success/failure indicators** (✅ ❌ ⚠️)

### Generation Feedback
During base generation, users see:
1. Requirements processing status
2. Prefab matching progress
3. Room placement updates
4. Decoration and corridor filling
5. Total time taken

## Known Issues & Important Notes

1. ✅ **Terrain decompression** - FIXED: Now successfully decoding 62,500 terrain tiles from binary data
2. **ASCII visualizations** don't render well for complex bases - use PNG visualizations instead
3. **Colonist detection** checks faction starting with "Faction_" and def containing "Human"
4. **Memory usage:** Large saves (100MB+) load entirely into memory - streaming parser exists but not used
5. **Script imports:** Some scripts missing path adjustment - add sys.path.insert if ModuleNotFoundError occurs

## Development Status

### Completed Features ✅
- **Phase 1: Parser & Models** - Full save file parsing with terrain decompression
- **Phase 2: Mod System Integration** - ModParser for XML definitions (`src/parser/mod_parser.py`)
- **Phase 3: Wave Function Collapse Generators**
  - Basic WFC (`src/generators/wfc_generator.py`)
  - Improved WFC (`src/generators/improved_wfc_generator.py`)
  - Hybrid prefab generator (`src/generators/hybrid_prefab_generator.py`)
  - Enhanced hybrid generator with multiple strategies
- **Phase 4: Natural Language Interface**
  - NLP parser for base requirements (`src/nlp/base_generator_nlp.py`)
  - Claude API integration (`src/ai/claude_base_designer.py`)
  - CLI interface (`rimworld_assistant.py`)
  - GUI interface with tkinter (`rimworld_assistant_gui.py`)

### Key Components

#### Requirements-Driven Generator (`src/generators/requirements_driven_generator.py`)
**NEW** - The core of the smart generation system:
- Matches NLP/Claude requirements to actual prefabs
- Scores prefabs based on size, aspect ratio, name matching
- Considers adjacency preferences (kitchen near dining, medical near entrance)
- Places high-priority rooms in optimal positions
- Fills remaining space with corridors and decorative elements

#### Prefab System
- **AlphaPrefabs Integration:** Loads and categorizes 100+ real RimWorld base designs
- **Prefab Analyzer:** Extracts rooms, patterns, and decorative elements
- **Hybrid Generator:** Can use complete prefabs, partial rooms, or just decorative elements
- **Usage Modes:** COMPLETE, PARTIAL, DECORATIVE, CONCEPTUAL

#### Visualization
- **Layered PNG:** Separate layers for each building type
- **Full Map:** Complete overview with detailed zoom
- **Interactive Viewer:** Navigate through generation layers

## File Organization

```
rimworldbuilder/
├── rimworld_assistant.py       # Main CLI interface
├── rimworld_assistant_gui.py   # GUI interface (tkinter)
├── src/
│   ├── parser/                 # Save file and mod parsing
│   ├── models/                 # Data models and entities
│   ├── generators/             # All generation algorithms (WFC, prefabs, hybrid)
│   ├── nlp/                    # Natural language processing
│   └── ai/                     # Claude API integration
├── scripts/                    # Utility and test scripts
├── data/
│   ├── saves/                  # Place .rws files here (gitignored)
│   └── AlphaPrefabs/          # Clone from GitHub for prefab support
└── docs/                       # Documentation
```
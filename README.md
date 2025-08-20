# RimWorld Base Assistant

An AI-powered tool for analyzing and optimizing RimWorld base layouts. Parses save files, understands mod configurations, and will generate optimized base designs using procedural generation techniques.

## Features

### Currently Implemented ✅
- **Save File Parser**: Reads and parses RimWorld .rws save files
- **Data Extraction**: Extracts buildings, colonists, zones, and items
- **Visualization**: Multiple visualization options for base layouts
  - PNG images with color-coded buildings
  - ASCII text representation
  - Full map overview and detailed views
- **Mod Support**: Detects and lists all active mods (176+ mods tested)

### In Development 🚧
- **Terrain Parsing**: Decompressing terrain grid data
- **Wave Function Collapse Generator**: Procedural base generation
- **Mod Definition Parser**: Extract building definitions from mods
- **Natural Language Interface**: Generate bases from text descriptions

## Installation

1. Ensure Python 3.11+ is installed
2. Install Poetry for dependency management
3. Clone this repository
4. Install dependencies:
```bash
poetry install
```

## Usage

### Parse a Save File
Place your RimWorld save file in `data/saves/` and run:
```bash
poetry run python scripts/test_parser.py
```

### Visualize Your Base
Generate various visualizations:
```bash
# Full map overview
poetry run python scripts/full_map_visualize.py

# ASCII text view
poetry run python scripts/text_visualize.py

# Detailed PIL visualization
poetry run python scripts/pil_visualize.py
```

## Project Structure
```
rimworld-base-assistant/
├── src/
│   ├── parser/          # Save file parsing
│   ├── models/          # Data models for game entities
│   ├── generators/      # Procedural generation algorithms
│   ├── nlp/            # Natural language processing
│   ├── evaluation/     # Base scoring functions
│   └── api/           # REST API (future)
├── data/
│   ├── saves/         # RimWorld save files (.rws)
│   └── mods/          # Mod definitions
├── scripts/           # Utility and visualization scripts
├── tests/            # Unit tests
└── docs/             # Documentation
```

## Save File Statistics

From the analyzed save file:
- **Map Size**: 250x250 tiles
- **Buildings**: 3,347 total
  - 972 Walls
  - 89 Furniture pieces
  - 22 Doors
  - 14 Power buildings
  - 2 Production buildings
- **Colonists**: 15
- **Items**: 5,751
- **Zones**: 11 (8 growing, 3 stockpile)
- **Active Mods**: 176

## Development Plan

Following the roadmap in `docs/DEV_PLAN.md`:

### Phase 1: Foundation ✅
- Save file parser
- Data models
- Basic visualization

### Phase 2: Mod System (Next)
- Parse mod definitions
- Build comprehensive building database

### Phase 3: Generation
- Wave Function Collapse implementation
- Constraint-based generation

### Phase 4: NLP Interface
- Natural language parsing
- Constraint translation

## Requirements

- Python 3.11+
- Poetry
- Dependencies: lxml, numpy, pydantic, PIL, matplotlib

## License

TBD

## Contributing

This project is in active development. See `docs/DEV_PLAN.md` for the development roadmap.
# RimWorld Base Assistant: Collaborative Development Plan with Claude Code

## Project Overview
We'll build a standalone RimWorld base-building assistant that parses save files, understands mod configurations, and generates optimized base layouts from natural language prompts. Claude Code will actively develop alongside you, browsing GitHub repositories for reference implementations and writing production-ready code.

## Key GitHub Repositories for Reference

### Essential RimWorld Tools to Study
- **[EnzoMartin/RimWorld-Save-Editor](https://github.com/EnzoMartin/RimWorld-Save-Editor)** - Node.js save file parser we'll examine for XML structure handling
- **[RimSort/RimSort](https://github.com/RimSort/RimSort)** - Python mod manager with comprehensive mod parsing logic
- **[CameronHudson8/rimworld-base-planner](https://github.com/CameronHudson8/rimworld-base-planner)** - TypeScript base optimizer using simulated annealing
- **[patchware-dev/Rimworld-XML-Data-Extractor](https://github.com/patchware-dev/Rimworld-XML-Data-Extractor)** - C# XML parsing specifically for RimWorld data

### Procedural Generation Libraries
- **[mxgmn/WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)** - Original WFC implementation we'll adapt for base generation
- **[ilmola/generator](https://github.com/ilmola/generator)** - C++11 procedural geometry library for optimization
- **[keon/awesome-nlp](https://github.com/keon/awesome-nlp)** - Curated NLP resources for parsing user prompts

## Development Plan Outline

### Phase 1: Foundation & Save File Parser (Week 1-2)
**Claude Code Tasks:**
- Set up Python project with Poetry for dependency management
- Browse RimWorld-Save-Editor repo to understand save structure
- Implement streaming XML parser using lxml for large files
- Create data models for buildings, terrain, colonists, and mods
- Write unit tests using sample .rws files

**Deliverables:**
- `rimworld_parser.py` - Core save file parser
- `models/` directory with dataclasses for game entities
- Test suite with 5+ real save files

### Phase 2: Mod System Integration (Week 3-4)
**Claude Code Tasks:**
- Study RimSort's mod detection and parsing logic
- Build mod definition parser for XML Def files
- Create building database from vanilla + modded content
- Implement inheritance resolution for mod definitions
- Handle Ideology DLC building restrictions

**Deliverables:**
- `mod_parser.py` - Extracts building definitions from mods
- `building_database.py` - Cached database of available structures
- Support for top 20 RimWorld mods

### Phase 3: Procedural Generation Engine (Week 5-7)
**Claude Code Tasks:**
- Port Wave Function Collapse from mxgmn's repo to Python
- Adapt WFC for RimWorld's grid system and building constraints
- Implement pathfinding for traffic flow evaluation
- Create constraint system for room adjacency rules
- Build fitness functions for defense, efficiency, aesthetics

**Deliverables:**
- `generators/wfc_generator.py` - Wave Function Collapse implementation
- `generators/genetic_algorithm.py` - Alternative GA-based generator
- `evaluation/` - Scoring functions for generated bases

### Phase 4: Natural Language Interface (Week 8-9)
**Claude Code Tasks:**
- Implement spaCy pipeline for intent extraction
- Create grammar for spatial relationships and priorities
- Build constraint translator from NLP to generation rules
- Add template system for common base types
- Implement feedback loop for ambiguous requests

**Deliverables:**
- `nlp/intent_parser.py` - Converts prompts to constraints
- `nlp/templates.py` - Pre-defined base archetypes
- Integration tests with 50+ example prompts

### Phase 5: User Interface & Visualization (Week 10-11)
**Claude Code Tasks:**
- Create Flask/FastAPI backend for generation API
- Build React frontend with base preview canvas
- Implement real-time generation with progress updates
- Add interactive constraint adjustment
- Create export format for planning mods

**Deliverables:**
- `api/` - REST API for generation requests
- `frontend/` - React application with visualization
- Docker configuration for easy deployment

### Phase 6: Optimization & Polish (Week 12)
**Claude Code Tasks:**
- Profile and optimize XML parsing for 100MB+ files
- Implement caching for mod definitions
- Add multi-threading for generation algorithms
- Create comprehensive error handling
- Write user documentation

**Deliverables:**
- Performance benchmarks
- User guide and API documentation
- GitHub Actions CI/CD pipeline

## Technical Stack & Architecture

### Core Technologies
```python
# requirements.txt structure
python = "^3.11"
lxml = "^4.9"          # Fast XML parsing
spacy = "^3.7"         # NLP processing
numpy = "^1.26"        # Array operations for WFC
fastapi = "^0.104"     # API framework
pydantic = "^2.5"      # Data validation
redis = "^5.0"         # Caching layer
pytest = "^7.4"        # Testing framework
```

### Project Structure
```
rimworld-base-assistant/
├── src/
│   ├── parser/           # Save file and mod parsing
│   ├── models/           # Data models
│   ├── generators/       # WFC, GA, other algorithms
│   ├── nlp/             # Natural language processing
│   ├── evaluation/      # Base scoring functions
│   └── api/            # REST API
├── frontend/           # React visualization
├── tests/              # Comprehensive test suite
├── data/               # Sample saves and mods
└── docs/               # Documentation
```

## Implementation Strategy with Claude Code

### Week 1-2: Parser Development
Claude Code will:
1. Browse EnzoMartin's save editor to understand XML structure
2. Implement incremental parser that handles 100MB+ files
3. Create comprehensive data models matching RimWorld's schema
4. Write extraction logic for map, buildings, and colonists

### Week 3-4: Mod System
Claude Code will:
1. Study RimSort's mod detection approach
2. Parse mod folders recursively for Def files
3. Build inheritance resolution for mod definitions
4. Create cached database of building properties

### Week 5-7: Generation Algorithms
Claude Code will:
1. Port WFC from C# to Python, optimizing for our use case
2. Define tile types and adjacency rules for RimWorld
3. Implement constraint propagation for user requirements
4. Add evaluation metrics for generated layouts

### Week 8-9: NLP Pipeline
Claude Code will:
1. Train spaCy model on RimWorld-specific terminology
2. Create intent classification for design goals
3. Build entity extraction for rooms and relationships
4. Implement constraint generation from parsed intents

### Week 10-11: User Interface
Claude Code will:
1. Design REST API with FastAPI
2. Create React components for base visualization
3. Implement WebSocket for real-time generation updates
4. Build export functionality for game compatibility

## Key Development Decisions

### Language Choice: Python with C++ Extensions
- **Python** for rapid development and NLP libraries
- **Cython/C++** extensions for performance-critical parsing
- Claude Code can seamlessly work with both languages

### Algorithm Priority
1. **Start with WFC** - Fastest, most predictable results
2. **Add Genetic Algorithm** - For multi-objective optimization
3. **Consider RL later** - If we need adaptive generation

### Data Handling Strategy
- **Streaming XML parsing** for large files
- **Redis caching** for mod definitions
- **SQLite** for persistent building database
- **Memory mapping** for files over 100MB

## Collaboration Workflow

### Daily Development Cycle
1. **Morning:** Review requirements, browse relevant GitHub repos
2. **Coding:** Claude Code implements features while explaining decisions
3. **Testing:** Write tests alongside implementation
4. **Documentation:** Update docs as we build

### Code Review Process
- Claude Code will explain architectural decisions
- Suggest optimizations based on profiling
- Reference similar implementations from GitHub
- Ensure code follows Python best practices

### Knowledge Transfer
- Claude Code will comment extensively
- Create architecture decision records (ADRs)
- Build runnable examples for each component
- Maintain development blog/notes

## Success Metrics

### Technical Goals
- Parse 100MB save files in <5 seconds
- Generate base layouts in <2 seconds
- Support top 50 RimWorld mods
- Handle natural language with 90% intent accuracy

### User Experience Goals
- One-click save file import
- Real-time preview of generated bases
- Export to planning mod formats
- Support for common base archetypes

## Next Steps

### Immediate Actions (Day 1)
1. Set up GitHub repository
2. Initialize Python project with Poetry
3. Claude Code browses RimWorld-Save-Editor for initial parser
4. Create first save file parser prototype
5. Test with your actual save files

### Week 1 Milestones
- Working XML parser for uncompressed saves
- Data models for core game entities
- Basic test suite with real save files
- Initial performance benchmarks

This collaborative approach leverages Claude Code's ability to browse repositories, understand existing implementations, and write production code while explaining decisions throughout the development process.
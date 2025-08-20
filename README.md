# RimWorld Base Assistant 🏰

An AI-powered tool for designing and generating RimWorld bases using real prefabs from the AlphaPrefabs mod, with natural language understanding and Claude AI integration.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Poetry** for dependency management
3. **AlphaPrefabs mod** (optional but recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/rimworldbuilder.git
cd rimworldbuilder

# 2. Install dependencies with Poetry
poetry install

# 3. (Optional) Clone AlphaPrefabs for full functionality
cd data
git clone https://github.com/juanosarg/AlphaPrefabs.git
cd ..

# 4. (Optional) Set up Claude API key for AI features
# Windows:
set ANTHROPIC_API_KEY=your-api-key-here
# Linux/Mac:
export ANTHROPIC_API_KEY=your-api-key-here
```

## 🎮 How to Launch

### Easy Launch (Windows/Linux/Mac)

#### Windows:
```bash
start.bat
```

#### Linux/Mac:
```bash
chmod +x start.sh
./start.sh
```

### Manual Launch

#### Option 1: Command-Line Interface (Recommended)
```bash
poetry run python rimworld_assistant.py
```

#### Option 2: Graphical User Interface
```bash
poetry run python rimworld_assistant_gui.py
```

## 📖 Using the Application

When you launch the CLI, you'll see this menu:

```
==================================================
                MAIN MENU
==================================================

1. Load Save File
2. Analyze Current Base
3. Generate from Natural Language
4. Generate with Prefab Anchors
5. Generate Enhanced Hybrid Base
6. AI-Designed Base (Claude)
7. Smart Generate (NLP → Prefabs)  ← RECOMMENDED
8. Visualize Last Generation
9. Interactive Layer Viewer
*. Export Base Design
0. Exit
```

### 🌟 Recommended: Smart Generate (Option 7)

This is the most advanced generation method:

1. **Describe your base in plain English:**
   ```
   > defensive base for 10 colonists with hospital and workshops
   ```

2. **The system will:**
   - Parse your requirements using NLP
   - Optionally use Claude AI for detailed planning
   - Match requirements to real RimWorld prefabs
   - Intelligently place prefabs with proper adjacency
   - Fill remaining space with corridors
   - Save a PNG visualization

## ✨ Features

### Currently Implemented ✅
- **Save File Parser**: Full .rws save file parsing with terrain decompression
- **Natural Language Interface**: Describe bases in plain English
- **Claude AI Integration**: Intelligent base planning with Claude API
- **Prefab System**: Uses 100+ real bases from AlphaPrefabs mod
- **Smart Generation**: Matches requirements to prefabs intelligently
- **Multiple Generators**: WFC, hybrid, prefab-based options
- **Real-time Progress**: Spinners, progress bars, and status updates
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🎯 Example Commands

### Basic Defense Base
```
defensive base for 8 colonists with killbox and medical bay
```

### Production Focus
```
efficient production base with 4 workshops, large storage, and minimal bedrooms
```

### Comfortable Living
```
spacious base for 12 colonists with recreation room, dining hall, and individual bedrooms
```

## 📁 Working with Save Files

Place your RimWorld save files (`.rws`) in:
```
data/saves/
```

The tool will automatically detect and list them when loading.

## 🖼️ Output

Generated bases are saved as PNG images:
- `smart_generated.png` - From Smart Generate
- `nlp_generated.png` - From Natural Language
- `ai_designed.png` - From AI Design

## ⚙️ Advanced Usage

### Using Scripts Directly

```bash
# Parse a save file
poetry run python scripts/test_parser.py

# Test visualization
poetry run python scripts/full_map_visualize.py

# Test progress indicators
poetry run python scripts/test_progress.py
```

## 🛠️ Troubleshooting

### "No module named 'src'"
Run from the project root directory, not from scripts/

### "AlphaPrefabs not found"
Clone AlphaPrefabs into data/ directory:
```bash
cd data
git clone https://github.com/juanosarg/AlphaPrefabs.git
```

### Unicode/Emoji Issues on Windows
The system automatically falls back to ASCII characters if Unicode isn't supported.

### Memory Issues with Large Saves
The tool loads entire saves into memory. For very large saves (100MB+), ensure you have sufficient RAM.

### Claude API Not Working
1. Ensure you have set the ANTHROPIC_API_KEY environment variable
2. Check you have API credits available
3. The system will fall back to non-AI generation if API is unavailable

## 📚 Project Structure

```
rimworldbuilder/
├── rimworld_assistant.py       # Main CLI interface
├── rimworld_assistant_gui.py   # GUI interface
├── src/
│   ├── parser/                 # Save file parsing
│   ├── models/                 # Data models
│   ├── generators/             # Base generation algorithms
│   │   ├── wfc_generator.py   # Wave Function Collapse
│   │   ├── hybrid_prefab_generator.py
│   │   └── requirements_driven_generator.py  # Smart generation
│   ├── nlp/                   # Natural language processing
│   ├── ai/                    # Claude AI integration
│   └── utils/                 # Progress indicators & symbols
├── scripts/                   # Utility scripts
├── data/
│   ├── saves/                # Place .rws files here
│   └── AlphaPrefabs/         # Clone from GitHub
└── docs/                     # Documentation

```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 Additional Resources

- `CLAUDE.md` - Technical documentation for developers
- `docs/DEV_PLAN.md` - Development roadmap
- [AlphaPrefabs Mod](https://github.com/juanosarg/AlphaPrefabs) - Source of prefab designs

---

**Made with ❤️ for the RimWorld community**
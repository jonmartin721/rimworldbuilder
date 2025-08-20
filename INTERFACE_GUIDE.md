# RimWorld Base Assistant - Interface Guide

## 🎨 Enhanced GUI Features (v2)

### Launching the Enhanced GUI

**Windows:**
```batch
start.bat
# Choose option 2
```

**Direct Launch:**
```bash
poetry run python rimworld_assistant_gui_v2.py
python rimworld_assistant.py --generate nlp --request "Create defensive base for 8 colonists" --output my_base.png
```

### Graphical Interface
```bash
# Run the GUI (requires tkinter)
python rimworld_assistant_gui.py
```

## CLI Interface Features

### Main Menu Options

1. **Load Save File** - Load your RimWorld .rws save file
2. **Analyze Current Base** - See statistics about your existing base
3. **Generate from Natural Language** - Describe what you want in plain English
4. **Generate with Prefab Anchors** - Use real AlphaPrefabs as foundation
5. **Generate Enhanced Hybrid Base** - Mix complete/partial/decorative prefabs
6. **AI-Designed Base** - Let Claude AI design your base
7. **Visualize Last Generation** - View the last generated base
8. **Interactive Layer Viewer** - Toggle building layers on/off
9. **Export Base Design** - Save as PNG, CSV, or JSON

### Natural Language Examples

```
"Create a defensive base for 10 colonists with killbox"
"Build a compact base for 5 people with workshop and hospital"
"Make an efficient base with large storage and recreation area"
"Design a fortress with multiple defensive layers"
```

### Generation Modes

- **Simple**: Basic WFC generation
- **Prefabs**: Uses complete AlphaPrefabs as anchors
- **Hybrid**: Mixes prefabs with procedural generation
- **Enhanced**: Uses prefabs at multiple scales (complete/partial/decorative)

## GUI Interface Features

### Main Window Layout

- **Left Panel**: Controls and options
  - Save file loader
  - Quick NLP generation
  - Generation options (size, density, mode)
  - Statistics display

- **Center Panel**: Visual preview of generated base
  - Real-time visualization
  - Color-coded room types

- **Right Panel**: Output log
  - Generation progress
  - Error messages
  - Statistics

### Quick Actions

1. **Load Save**: File → Load Save File
2. **Quick Generate**: Type description and click "Generate from Text"
3. **Advanced Generate**: Set options and click "Generate Base"
4. **Export**: File → Export Base

### Keyboard Shortcuts

- `Ctrl+O`: Load save file
- `Ctrl+S`: Export base
- `Ctrl+Q`: Exit

## Color Legend

- **Gray**: Walls
- **Brown**: Doors
- **Dark Gray**: Corridors
- **Blue**: Bedrooms
- **Orange**: Kitchen/Dining
- **Green**: Storage
- **Yellow**: Workshop/Production
- **Purple**: Recreation
- **Red**: Medical
- **Light Yellow**: Power

## Tips for Best Results

### Natural Language Generation
- Be specific about colonist count
- Mention priorities (defense, efficiency, comfort)
- Include special requirements (killbox, hospital, throne room)

### Prefab Generation
- Start with 3-5 prefabs for balanced bases
- Use 0.3-0.5 density for good coverage
- Mix bedroom, kitchen, and workshop categories

### Enhanced Hybrid
- Use Complete + Partial for authentic-looking bases
- Add Decorative mode for furniture details
- Conceptual mode applies design patterns (symmetry, wall thickness)

## Advanced Features

### Using Your Save's Buildable Areas
The system automatically detects bridges from your save and can generate bases that fit your buildable areas.

### AI Design (Requires API Key)
```bash
export ANTHROPIC_API_KEY="your-key-here"
python rimworld_assistant.py
```

Then select option 6 for AI-designed bases.

### Batch Generation
```bash
# Generate multiple variations
for i in {1..5}; do
    python rimworld_assistant.py --generate hybrid --output "base_variant_$i.png"
done
```

## Troubleshooting

### "AlphaPrefabs not found"
Clone the AlphaPrefabs repository:
```bash
git clone https://github.com/juanosarg/AlphaPrefabs.git data/AlphaPrefabs
```

### GUI not working
Install tkinter (usually included with Python):
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows/Mac - Usually pre-installed
```

### Slow generation
- Reduce base size (try 40x40 or 60x60)
- Lower prefab density (0.2-0.3)
- Use Simple mode for faster generation

## Examples

### Defensive Base
```
"Create a heavily defended base for 12 colonists with:
- Central killbox with multiple fallback positions
- Hospital near entrance for quick rescue
- Armory adjacent to defensive positions
- Individual bedrooms in secure inner area
- Production facilities protected but accessible"
```

### Efficient Production Base
```
"Build an efficient production base for 8 colonists with:
- Clustered workshops for material efficiency
- Central storage between workshops and stockpiles
- Kitchen adjacent to freezer and dining room
- Minimal walking distances
- Separate clean and dirty work areas"
```

### Comfortable Colony
```
"Design a comfortable base for 6 colonists with:
- Spacious individual bedrooms
- Large recreation room with various joy sources
- Beautiful dining room
- Garden areas for beauty
- Separate guest quarters"
```
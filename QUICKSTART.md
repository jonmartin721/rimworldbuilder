# 🚀 RimWorld Base Assistant - Quick Start Guide

## 5-Minute Setup

### 1. Install Prerequisites
- Python 3.8+ 
- Poetry (`pip install poetry`)

### 2. Clone & Setup
```bash
git clone https://github.com/yourusername/rimworldbuilder.git
cd rimworldbuilder
poetry install
```

### 3. Launch
**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
./start.sh
```

## Your First Base Generation

### Step 1: Launch the CLI
Choose option 1 when prompted

### Step 2: Select "Smart Generate" (Option 7)
This is the best generation method

### Step 3: Describe Your Base
Type something like:
```
defensive base for 8 colonists with medical bay
```

### Step 4: Wait for Generation
Watch the progress indicators:
- ⠋⠙⠹⠸ Processing your requirements
- Progress bars for placement
- ✅ Success indicators

### Step 5: View Your Base
Open `smart_generated.png` to see your generated base!

## Common Commands

| What You Want | What to Type |
|--------------|--------------|
| Small starter base | `compact base for 5 colonists` |
| Defensive fortress | `defensive base with killbox and turrets` |
| Production facility | `workshop base with 4 production rooms` |
| Research outpost | `research base with labs and minimal defense` |

## Tips

1. **Add AlphaPrefabs** for better results:
   ```bash
   cd data
   git clone https://github.com/juanosarg/AlphaPrefabs.git
   ```

2. **Use Claude AI** for smarter designs:
   ```bash
   set ANTHROPIC_API_KEY=your-key-here
   ```

3. **Load Your Save** (optional):
   - Place `.rws` files in `data/saves/`
   - Choose option 1 to load

## Need Help?

- Full documentation: See `README.md`
- Having issues? Check the Troubleshooting section in README.md

---

**Ready to build amazing RimWorld bases? Launch the app and try option 7!** 🏰
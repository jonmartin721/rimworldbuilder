#!/usr/bin/env python3
"""
Quick test to verify live preview works during training
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_training_gui import MLTrainingGUI

def main():
    print("=" * 60)
    print("Live Training Preview Test")
    print("=" * 60)
    print()
    print("Testing visual previews during training:")
    print()
    print("1. Click 'Test Preview' button to generate previews immediately")
    print("   - Should show 3 bases in Best/Worst/Latest panels")
    print()
    print("2. Start training to see previews update EVERY epoch")
    print("   - Previews will update automatically each epoch")
    print("   - Default feedback interval is now 10 epochs")
    print()
    print("3. The previews work even without a trained model")
    print("   - Uses rule-based generation initially")
    print("   - Switches to AI model once training progresses")
    print()
    
    # Launch GUI
    app = MLTrainingGUI()
    app.run()

if __name__ == "__main__":
    main()
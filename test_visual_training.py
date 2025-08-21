#!/usr/bin/env python3
"""
Test script for visual training progress features
"""

import sys
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_training_gui import MLTrainingGUI

def main():
    """Run the ML training GUI with visual progress"""
    print("=" * 60)
    print("Testing Visual Training Progress Features")
    print("=" * 60)
    print()
    print("New features to test:")
    print("1. Live preview of best/worst/latest samples during training")
    print("2. Side-by-side feedback dialog for multiple samples")
    print("3. Visual progress updates every 5 epochs")
    print("4. Fixed index out of bounds error")
    print()
    print("To test:")
    print("- Start training and watch the 'Live Training Preview' section")
    print("- Change feedback interval during training (should not crash)")
    print("- Click 'Test Side-by-Side' to test new feedback dialog")
    print()
    
    # Launch the GUI
    gui = MLTrainingGUI()
    gui.run()

if __name__ == "__main__":
    main()
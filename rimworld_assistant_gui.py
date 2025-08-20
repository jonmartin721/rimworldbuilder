#!/usr/bin/env python3
"""
RimWorld Base Assistant - Professional GUI
Direct launch of the professional interface with modern design.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the professional GUI directly"""
    # Launch the v3 professional GUI
    script_path = Path(__file__).parent / "rimworld_assistant_gui_v3.py"
    subprocess.run([sys.executable, str(script_path)])

if __name__ == "__main__":
    main()
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lxml import etree
import re

def find_heavybridge_context():
    tree = etree.parse('data/saves/Autosave-2.rws')
    root = tree.getroot()
    game = root.find('game')
    maps = game.find('maps')
    first_map = maps.find('li')
    
    # Convert to string to search
    map_str = etree.tostring(first_map, encoding='unicode')
    
    # Find contexts where HeavyBridge appears (but not Frame_HeavyBridge)
    pattern = r'<(\w+)>([^<]*HeavyBridge[^<]*)</\1>'
    matches = re.findall(pattern, map_str)
    
    # Count different contexts
    contexts = {}
    for tag, content in matches:
        if 'Frame_HeavyBridge' not in content:
            contexts[tag] = contexts.get(tag, 0) + 1
    
    print("HeavyBridge (non-Frame) appears in these XML tags:")
    for tag, count in sorted(contexts.items(), key=lambda x: x[1], reverse=True):
        print(f"  <{tag}>: {count} times")
    
    # Check if it's in terrain or underGrid
    terrain_grid = first_map.find('terrainGrid')
    if terrain_grid is not None:
        data = terrain_grid.find('data')
        if data is not None and data.text:
            if 'HeavyBridge' in data.text:
                print("\nHeavyBridge found in terrainGrid!")
                # Show a sample
                idx = data.text.find('HeavyBridge')
                print(f"  Sample: ...{data.text[max(0,idx-20):idx+30]}...")
    
    under_grid = first_map.find('underGrid')
    if under_grid is not None:
        data = under_grid.find('data')
        if data is not None and data.text:
            if 'HeavyBridge' in data.text:
                print("\nHeavyBridge found in underGrid!")
                idx = data.text.find('HeavyBridge')
                print(f"  Sample: ...{data.text[max(0,idx-20):idx+30]}...")

find_heavybridge_context()
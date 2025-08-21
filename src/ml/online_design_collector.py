"""
Collects RimWorld base designs from online sources for training data.
Scrapes designs from Reddit, Steam Workshop descriptions, and forums.
"""

import requests
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class OnlineDesign:
    """Represents a base design found online"""
    source: str  # e.g., "reddit", "steam", "forum"
    url: str
    title: str
    description: str
    layout_text: Optional[str] = None  # ASCII representation if available
    score: int = 0  # Upvotes, likes, etc.
    colonist_count: Optional[int] = None
    tags: List[str] = None
    

class OnlineDesignCollector:
    """Collects base designs from various online sources"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.designs = []
        
    def collect_from_reddit(self, subreddit: str = "RimWorld", limit: int = 100) -> List[OnlineDesign]:
        """Collect base designs from Reddit"""
        designs = []
        
        # Reddit API endpoint (using JSON)
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': 'base layout OR base design OR colony layout',
            'limit': limit,
            'sort': 'top',
            't': 'all'
        }
        
        headers = {'User-Agent': 'RimWorldBaseCollector/1.0'}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    # Filter for posts likely containing base designs
                    if any(word in post_data['title'].lower() 
                           for word in ['base', 'layout', 'colony', 'design']):
                        
                        design = OnlineDesign(
                            source='reddit',
                            url=f"https://reddit.com{post_data['permalink']}",
                            title=post_data['title'],
                            description=post_data.get('selftext', '')[:500],
                            score=post_data['ups'],
                            tags=self._extract_tags(post_data['title'])
                        )
                        
                        # Try to extract colonist count from title/description
                        colonist_match = re.search(r'(\d+)\s*(?:colonist|pawn)', 
                                                 post_data['title'] + post_data.get('selftext', ''), 
                                                 re.IGNORECASE)
                        if colonist_match:
                            design.colonist_count = int(colonist_match.group(1))
                        
                        designs.append(design)
                        
                logger.info(f"Collected {len(designs)} designs from Reddit")
                
        except Exception as e:
            logger.error(f"Error collecting from Reddit: {e}")
            
        return designs
    
    def collect_from_steam_workshop(self, app_id: int = 294100, limit: int = 50) -> List[OnlineDesign]:
        """Collect base designs from Steam Workshop"""
        designs = []
        
        # Steam Workshop API
        url = "https://api.steampowered.com/IPublishedFileService/QueryFiles/v1/"
        params = {
            'key': 'YOUR_STEAM_API_KEY',  # Need to get from Steam
            'appid': app_id,  # RimWorld
            'search_text': 'base layout prefab',
            'return_metadata': True,
            'numperpage': limit
        }
        
        # For now, return empty since we need API key
        logger.info("Steam Workshop collection requires API key")
        return designs
    
    def parse_ascii_layout(self, text: str) -> Optional[np.ndarray]:
        """Parse ASCII representation of a base layout"""
        # Look for ASCII art patterns
        lines = text.split('\n')
        
        # Find lines that look like base layouts (contain walls, doors, etc.)
        layout_lines = []
        for line in lines:
            if any(char in line for char in ['#', '+', 'D', 'B', 'W', '|', '-']):
                layout_lines.append(line)
        
        if len(layout_lines) > 5:  # At least 5 lines for a meaningful layout
            # Convert to numpy array (simplified)
            max_width = max(len(line) for line in layout_lines)
            layout = np.zeros((len(layout_lines), max_width), dtype=int)
            
            char_map = {
                '#': 1,  # Wall
                '+': 2,  # Door
                'D': 2,  # Door
                'B': 3,  # Bed
                'W': 7,  # Workbench
                '.': 10, # Floor
                ' ': 0   # Empty
            }
            
            for i, line in enumerate(layout_lines):
                for j, char in enumerate(line[:max_width]):
                    layout[i, j] = char_map.get(char, 0)
                    
            return layout
            
        return None
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        tags = []
        
        tag_patterns = [
            'mountain', 'flat', 'river', 'coast', 'desert', 'ice sheet',
            'tribal', 'industrial', 'spacer', 'medieval',
            'killbox', 'peaceful', 'defense', 'efficient',
            'early game', 'mid game', 'late game', 'end game'
        ]
        
        text_lower = text.lower()
        for pattern in tag_patterns:
            if pattern in text_lower:
                tags.append(pattern)
                
        return tags
    
    def save_collected_designs(self, filename: str = "online_designs.json"):
        """Save collected designs to file"""
        filepath = self.cache_dir / filename
        
        designs_data = []
        for design in self.designs:
            design_dict = {
                'source': design.source,
                'url': design.url,
                'title': design.title,
                'description': design.description,
                'score': design.score,
                'colonist_count': design.colonist_count,
                'tags': design.tags
            }
            designs_data.append(design_dict)
        
        with open(filepath, 'w') as f:
            json.dump(designs_data, f, indent=2)
            
        logger.info(f"Saved {len(designs_data)} designs to {filepath}")
    
    def load_cached_designs(self, filename: str = "online_designs.json") -> List[OnlineDesign]:
        """Load previously collected designs"""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                designs_data = json.load(f)
                
            designs = []
            for data in designs_data:
                design = OnlineDesign(
                    source=data['source'],
                    url=data['url'],
                    title=data['title'],
                    description=data['description'],
                    score=data.get('score', 0),
                    colonist_count=data.get('colonist_count'),
                    tags=data.get('tags', [])
                )
                designs.append(design)
                
            logger.info(f"Loaded {len(designs)} cached designs")
            return designs
            
        return []
    
    def collect_all(self) -> List[OnlineDesign]:
        """Collect from all available sources"""
        all_designs = []
        
        # Try loading cache first
        cached = self.load_cached_designs()
        if cached:
            logger.info(f"Using {len(cached)} cached designs")
            return cached
        
        # Collect from Reddit
        logger.info("Collecting from Reddit...")
        reddit_designs = self.collect_from_reddit()
        all_designs.extend(reddit_designs)
        time.sleep(2)  # Rate limiting
        
        # Could add more sources here
        # steam_designs = self.collect_from_steam_workshop()
        # all_designs.extend(steam_designs)
        
        self.designs = all_designs
        
        # Save for future use
        if all_designs:
            self.save_collected_designs()
        
        return all_designs


def integrate_online_designs_into_training(data_dir: Path):
    """Integrate online designs into training dataset"""
    collector = OnlineDesignCollector(data_dir)
    designs = collector.collect_all()
    
    # Convert designs to training examples (simplified)
    training_examples = []
    for design in designs:
        # Here we'd convert the design info into actual training data
        # For now, just log what we found
        logger.info(f"Found design: {design.title} ({design.score} upvotes)")
        
        if design.layout_text:
            layout = collector.parse_ascii_layout(design.layout_text)
            if layout is not None:
                training_examples.append({
                    'layout': layout,
                    'metadata': {
                        'source': design.source,
                        'colonists': design.colonist_count,
                        'tags': design.tags
                    }
                })
    
    logger.info(f"Converted {len(training_examples)} designs to training examples")
    return training_examples
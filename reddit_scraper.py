#!/usr/bin/env python3
"""
Reddit RimWorld Base Design Scraper
Collects base design images from Reddit for ML training data

Standalone tool for collecting training data from Reddit, separate from main training GUI.
"""

import time
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import re
import argparse
from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class RedditPost:
    """Reddit post data"""
    id: str
    title: str
    url: str
    image_url: Optional[str]
    score: int
    num_comments: int
    created_utc: float
    author: str
    subreddit: str
    permalink: str
    
    def get_filename(self) -> str:
        """Generate filename for this post"""
        # Clean title for filename
        clean_title = re.sub(r'[^\w\s-]', '', self.title.lower())
        clean_title = re.sub(r'[-\s]+', '-', clean_title)[:50]
        return f"{self.subreddit}_{self.id}_{clean_title}.png"


class RedditScraper:
    """Scrapes Reddit for RimWorld base designs"""
    
    def __init__(self, output_dir: str = "data/reddit_bases"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Reddit API configuration
        self.headers = {
            'User-Agent': 'RimWorldBaseCollector/1.0 (ML Training Data Collection)'
        }
        
        # Track processed posts
        self.processed_file = self.output_dir / "processed_posts.json"
        self.processed_ids = self.load_processed_ids()
        
    def load_processed_ids(self) -> set:
        """Load already processed post IDs"""
        if self.processed_file.exists():
            with open(self.processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_ids(self):
        """Save processed post IDs"""
        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_ids), f)
    
    def search_subreddit(self, subreddit: str, query: str = "", sort: str = "top", 
                        time_filter: str = "all", limit: int = 100) -> List[RedditPost]:
        """Search a subreddit for posts"""
        posts = []
        
        # Build URL
        if query:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': query,
                'restrict_sr': 'on',
                'sort': sort,
                't': time_filter,
                'limit': min(limit, 100)
            }
        else:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {
                't': time_filter,
                'limit': min(limit, 100)
            }
        
        try:
            print(f"Fetching from r/{subreddit} (sort={sort}, time={time_filter})...")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for child in data['data']['children']:
                post_data = child['data']
                
                # Check if post has an image
                image_url = self.extract_image_url(post_data)
                if image_url:
                    post = RedditPost(
                        id=post_data['id'],
                        title=post_data['title'],
                        url=post_data['url'],
                        image_url=image_url,
                        score=post_data['score'],
                        num_comments=post_data['num_comments'],
                        created_utc=post_data['created_utc'],
                        author=post_data.get('author', '[deleted]'),
                        subreddit=post_data['subreddit'],
                        permalink=f"https://reddit.com{post_data['permalink']}"
                    )
                    posts.append(post)
            
            print(f"Found {len(posts)} posts with images")
            
        except Exception as e:
            print(f"Error fetching from Reddit: {e}")
        
        return posts
    
    def extract_image_url(self, post_data: dict) -> Optional[str]:
        """Extract image URL from post data"""
        url = post_data.get('url', '')
        
        # Direct image links
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            return url
        
        # Reddit image hosting
        if 'i.redd.it' in url:
            return url
        
        # Imgur links
        if 'imgur.com' in url:
            # Convert gallery links to direct image links
            if '/a/' in url or '/gallery/' in url:
                return None  # Skip galleries for now
            if not any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return url + '.png'  # Try adding extension
            return url
        
        # Check preview images
        if 'preview' in post_data:
            try:
                images = post_data['preview'].get('images', [])
                if images:
                    # Get the source image
                    source = images[0].get('source', {})
                    image_url = source.get('url', '').replace('&amp;', '&')
                    if image_url:
                        return image_url
            except Exception:
                pass
        
        return None
    
    def download_image(self, url: str, filepath: Path) -> bool:
        """Download an image from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify it's a valid image
            try:
                img = Image.open(filepath)
                img.verify()
                return True
            except Exception:
                filepath.unlink()  # Delete invalid file
                return False
                
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def filter_base_designs(self, img: Image.Image) -> bool:
        """
        Filter images to find likely base designs
        Returns True if image appears to be a RimWorld base screenshot
        """
        # Convert to numpy array
        img_array = np.array(img)
        
        # Check image dimensions
        height, width = img_array.shape[:2]
        
        # More lenient aspect ratio (RimWorld can be any aspect)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            return False
        
        # Check image size (filter out tiny thumbnails)
        # Be more lenient - some people post smaller screenshots
        if width < 600 or height < 400:
            return False
        
        # Check if it's mostly UI (very bright or very dark)
        if len(img_array.shape) == 3:  # Color image
            # Convert to grayscale for brightness check
            gray = img_array.mean(axis=2)
            mean_brightness = gray.mean()
            
            # Skip if too dark (likely not a game screenshot)
            if mean_brightness < 30:
                return False
            
            # Skip if too bright (likely text/meme)
            if mean_brightness > 240:
                return False
        
        # For now, be more permissive to collect more data
        # Manual filtering can be done later
        return True
    
    def process_post(self, post: RedditPost) -> bool:
        """Process a single post"""
        # Skip if already processed
        if post.id in self.processed_ids:
            return False
        
        # Download image
        image_path = self.images_dir / post.get_filename()
        
        if not post.image_url:
            return False
        
        print(f"Processing: {post.title[:50]}...")
        
        if self.download_image(post.image_url, image_path):
            # Verify it looks like a base design
            try:
                img = Image.open(image_path)
                
                if not self.filter_base_designs(img):
                    print("  Filtered out (not a base design)")
                    img.close()  # Close image before deleting
                    image_path.unlink()
                    return False
                
                # Close image after checking
                img.close()
                
                # Save metadata
                metadata = {
                    'id': post.id,
                    'title': post.title,
                    'url': post.url,
                    'image_url': post.image_url,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created': datetime.fromtimestamp(post.created_utc).isoformat(),
                    'author': post.author,
                    'subreddit': post.subreddit,
                    'permalink': post.permalink,
                    'image_file': post.get_filename(),
                    'processed': datetime.now().isoformat()
                }
                
                metadata_path = self.metadata_dir / f"{post.id}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Mark as processed
                self.processed_ids.add(post.id)
                print(f"  [SUCCESS] Saved: {post.get_filename()}")
                return True
                
            except Exception as e:
                print(f"  Error processing image: {e}")
                try:
                    if image_path.exists():
                        image_path.unlink()
                except Exception:
                    pass  # Ignore file deletion errors
                return False
        
        return False
    
    def scrape_rimworld_bases(self, 
                            subreddits: List[str] = None,
                            search_terms: List[str] = None,
                            max_posts: int = 100,
                            time_filter: str = "all"):
        """Main scraping function"""
        
        if subreddits is None:
            # RimWorld subreddits where base designs are posted
            # RimWorldPorn is specifically for colony screenshots
            subreddits = ["RimWorld", "RimWorldPorn"]
        
        if search_terms is None:
            # Search terms based on common post titles and flairs
            # Focus on terms that typically have base screenshots
            search_terms = ["flair:colony_showcase", "flair:art", 
                          "my base", "my colony", "base tour", 
                          "year colony", "tile base", "mountain base",
                          "ice sheet base", "desert base", "sea ice",
                          "megabase", "modded base", "vanilla base"]
        
        all_posts = []
        
        # Search each subreddit
        for subreddit in subreddits:
            print(f"\n{'='*60}")
            print(f"Searching r/{subreddit}")
            print(f"{'='*60}")
            
            # Get top posts (no search terms needed for RimWorldPorn)
            posts = self.search_subreddit(subreddit, "", "top", time_filter, max_posts)
            all_posts.extend(posts)
            
            # Only search with keywords for main RimWorld subreddit
            # RimWorldPorn is already focused on colony screenshots
            if subreddit == "RimWorld":
                for term in search_terms[:5]:  # Limit to first 5 terms to avoid rate limiting
                    time.sleep(2)  # Rate limiting
                    posts = self.search_subreddit(subreddit, term, "top", time_filter, max_posts // 5)
                    all_posts.extend(posts)
        
        # Remove duplicates
        unique_posts = {p.id: p for p in all_posts}.values()
        print(f"\nTotal unique posts found: {len(unique_posts)}")
        
        # Process posts
        successful = 0
        for post in unique_posts:
            if self.process_post(post):
                successful += 1
                time.sleep(1)  # Rate limiting
        
        # Save processed IDs
        self.save_processed_ids()
        
        print(f"\n{'='*60}")
        print("Scraping complete!")
        print(f"Successfully downloaded: {successful} images")
        print(f"Total processed: {len(self.processed_ids)} posts")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        return successful
    
    def generate_dataset_summary(self):
        """Generate a summary of the collected dataset"""
        images = list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg"))
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        print("\nDataset Summary:")
        print(f"  Images: {len(images)}")
        print(f"  Metadata files: {len(metadata_files)}")
        
        if metadata_files:
            # Analyze metadata
            total_score = 0
            total_comments = 0
            subreddit_counts = {}
            
            for meta_file in metadata_files:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    total_score += data['score']
                    total_comments += data['num_comments']
                    subreddit = data['subreddit']
                    subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
            
            print(f"  Average score: {total_score / len(metadata_files):.1f}")
            print(f"  Average comments: {total_comments / len(metadata_files):.1f}")
            print("  Subreddit distribution:")
            for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    r/{sub}: {count}")
        
        # Save summary
        summary_file = self.output_dir / "dataset_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("RimWorld Base Design Dataset\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Images: {len(images)}\n")
            f.write(f"Metadata files: {len(metadata_files)}\n")
            f.write(f"Processed posts: {len(self.processed_ids)}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Scrape Reddit for RimWorld base designs")
    parser.add_argument("--output", "-o", default="data/reddit_bases", 
                       help="Output directory for images and metadata")
    parser.add_argument("--subreddits", "-s", nargs="+", 
                       default=["RimWorld", "RimWorldPorn"],
                       help="Subreddits to search (default: r/RimWorld, r/RimWorldPorn)")
    parser.add_argument("--max-posts", "-m", type=int, default=100,
                       help="Maximum posts to fetch per subreddit")
    parser.add_argument("--time", "-t", choices=["all", "year", "month", "week", "day"],
                       default="year", help="Time filter for posts (default: year)")
    parser.add_argument("--keywords", "-k", nargs="+",
                       help="Additional search keywords")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("RimWorld Reddit Base Design Scraper")
    print(f"{'='*60}\n")
    
    scraper = RedditScraper(args.output)
    
    # Run scraping
    search_terms = ["base", "colony", "layout", "design", "screenshot"]
    if args.keywords:
        search_terms.extend(args.keywords)
    
    scraper.scrape_rimworld_bases(
        subreddits=args.subreddits,
        search_terms=search_terms,
        max_posts=args.max_posts,
        time_filter=args.time
    )
    
    # Generate summary
    scraper.generate_dataset_summary()
    
    print("\nDone! You can now use the collected images for ML training.")
    print("To use in training, load the dataset from the training GUI.")


if __name__ == "__main__":
    main()
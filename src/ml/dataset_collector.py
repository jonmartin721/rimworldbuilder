"""
Dataset collector for training the base generation model.
Collects bases from multiple sources:
1. AlphaPrefabs XML files
2. Online base images and descriptions
3. User feedback on generated bases
4. Synthetic data from rule-based generators
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import requests
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import pickle
from datetime import datetime

from src.generators.realistic_base_generator import CellType
from src.ml.base_gan_model import BaseRequirements

logger = logging.getLogger(__name__)


@dataclass
class BaseExample:
    """A single base example with metadata"""
    layout: np.ndarray  # 2D array of cell types
    requirements: BaseRequirements
    quality_scores: Dict[str, float]  # efficiency, beauty, defense, etc.
    source: str  # Where this example came from
    description: str  # Text description
    
    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to tensors for training"""
        # Convert layout to one-hot encoding
        layout_tensor = torch.from_numpy(self.layout).long()
        
        # Get requirements tensor
        req_tensor = self.requirements.to_tensor()
        
        # Quality scores tensor
        quality_tensor = torch.tensor([
            self.quality_scores.get('efficiency', 0.5),
            self.quality_scores.get('beauty', 0.5),
            self.quality_scores.get('defense', 0.5),
            self.quality_scores.get('connectivity', 0.5),
            self.quality_scores.get('space_usage', 0.5),
        ], dtype=torch.float32)
        
        return layout_tensor, req_tensor, quality_tensor


class AlphaPrefabsParser:
    """Parse AlphaPrefabs XML files to extract base layouts"""
    
    CELL_MAPPING = {
        'Wall': CellType.WALL,
        'Door': CellType.DOOR,
        'Bed': CellType.BED,
        'Table': CellType.TABLE,
        'Chair': CellType.CHAIR,
        'DiningChair': CellType.CHAIR,
        'EndTable': CellType.ENDTABLE,
        'Dresser': CellType.DRESSER,
        'PlantPot': CellType.PLANT_POT,
        'TorchLamp': CellType.TORCH,
        'Campfire': CellType.TORCH,
        'Battery': CellType.BATTERY,
        'SolarGenerator': CellType.SOLAR_PANEL,
        'WindTurbine': CellType.WIND_TURBINE,
        'Cooler': CellType.COOLER,
        'Heater': CellType.HEATER,
        'Sandbag': CellType.SANDBAG,
        'Turret': CellType.TURRET,
    }
    
    def __init__(self, prefabs_dir: Path):
        self.prefabs_dir = Path(prefabs_dir)
        
    def parse_layout_def(self, xml_path: Path) -> List[BaseExample]:
        """Parse a single layout XML file"""
        examples = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for layout_def in root.findall('.//KCSG.StructureLayoutDef'):
                def_name = layout_def.find('defName').text
                
                # Parse layout grid
                layouts = layout_def.find('layouts')
                if layouts is None:
                    continue
                    
                for layout in layouts:
                    grid = self._parse_grid(layout)
                    if grid is not None:
                        # Infer requirements from layout
                        requirements = self._infer_requirements(grid, def_name)
                        
                        # Estimate quality scores
                        quality = self._estimate_quality(grid)
                        
                        example = BaseExample(
                            layout=grid,
                            requirements=requirements,
                            quality_scores=quality,
                            source=f"AlphaPrefabs/{xml_path.name}",
                            description=def_name
                        )
                        examples.append(example)
                        
        except Exception as e:
            logger.error(f"Error parsing {xml_path}: {e}")
            
        return examples
    
    def _parse_grid(self, layout_element) -> Optional[np.ndarray]:
        """Parse grid from XML layout element"""
        rows = []
        max_width = 0
        
        for row in layout_element:
            if row.tag == 'li':
                cells = row.text.split(',') if row.text else []
                max_width = max(max_width, len(cells))
                rows.append(cells)
        
        if not rows:
            return None
            
        # Create grid
        grid = np.zeros((len(rows), max_width), dtype=int)
        
        for y, row in enumerate(rows):
            for x, cell in enumerate(row):
                cell_type = self._parse_cell(cell.strip())
                grid[y, x] = cell_type.value
                
        return grid
    
    def _parse_cell(self, cell_str: str) -> CellType:
        """Parse cell string to CellType"""
        if cell_str == '.' or not cell_str:
            return CellType.EMPTY
            
        # Check for known patterns
        for pattern, cell_type in self.CELL_MAPPING.items():
            if pattern in cell_str:
                return cell_type
                
        # Check for floor patterns
        if 'Floor' in cell_str or 'Carpet' in cell_str:
            return CellType.FLOOR
            
        # Default to wall for unknown
        return CellType.WALL
    
    def _infer_requirements(self, grid: np.ndarray, name: str) -> BaseRequirements:
        """Infer requirements from grid layout"""
        # Count different cell types
        unique, counts = np.unique(grid, return_counts=True)
        cell_counts = dict(zip(unique, counts))
        
        # Estimate requirements
        num_beds = cell_counts.get(CellType.BED.value, 0)
        has_kitchen = cell_counts.get(CellType.STOVE.value, 0) > 0
        has_hospital = cell_counts.get(CellType.MEDICAL_BED.value, 0) > 0
        has_recreation = 'recreation' in name.lower() or 'rec' in name.lower()
        
        # Estimate defense level
        num_turrets = cell_counts.get(CellType.TURRET.value, 0)
        num_sandbags = cell_counts.get(CellType.SANDBAG.value, 0)
        defense_level = min(1.0, (num_turrets * 0.2 + num_sandbags * 0.05))
        
        return BaseRequirements(
            num_colonists=max(3, num_beds),
            num_bedrooms=num_beds,
            num_workshops=2,  # Default estimate
            has_kitchen=has_kitchen,
            has_hospital=has_hospital,
            has_recreation=has_recreation,
            defense_level=defense_level,
            beauty_preference=0.5,  # Default
            efficiency_preference=0.7,  # Default
            size_constraint=grid.shape[::-1]  # (width, height)
        )
    
    def _estimate_quality(self, grid: np.ndarray) -> Dict[str, float]:
        """Estimate quality scores from grid"""
        # Calculate various metrics
        total_cells = grid.size
        empty_cells = np.sum(grid == CellType.EMPTY.value)
        wall_cells = np.sum(grid == CellType.WALL.value)
        
        # Space usage efficiency
        space_usage = 1.0 - (empty_cells / total_cells)
        
        # Connectivity (rough estimate based on door placement)
        num_doors = np.sum(grid == CellType.DOOR.value)
        connectivity = min(1.0, num_doors * 0.1)
        
        # Beauty (based on decorative elements)
        decorative = np.sum(grid == CellType.PLANT_POT.value)
        beauty = min(1.0, decorative * 0.2 + 0.3)
        
        # Defense
        defensive = np.sum((grid == CellType.TURRET.value) | (grid == CellType.SANDBAG.value))
        defense = min(1.0, defensive * 0.1)
        
        # Efficiency (based on layout compactness)
        efficiency = space_usage * 0.5 + connectivity * 0.5
        
        return {
            'efficiency': efficiency,
            'beauty': beauty,
            'defense': defense,
            'connectivity': connectivity,
            'space_usage': space_usage,
        }
    
    def collect_all(self) -> List[BaseExample]:
        """Collect all examples from AlphaPrefabs"""
        examples = []
        
        # Find all layout XML files
        layout_files = list(self.prefabs_dir.glob('**/Layouts_*.xml'))
        
        for xml_file in layout_files:
            file_examples = self.parse_layout_def(xml_file)
            examples.extend(file_examples)
            logger.info(f"Collected {len(file_examples)} examples from {xml_file.name}")
            
        return examples


class UserFeedbackCollector:
    """Collect user feedback on generated bases"""
    
    def __init__(self, feedback_file: Path):
        self.feedback_file = Path(feedback_file)
        self.feedback_data = []
        
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                self.feedback_data = json.load(f)
    
    def add_feedback(self, layout: np.ndarray, requirements: BaseRequirements,
                     rating: float, comments: str = ""):
        """Add user feedback for a generated base"""
        feedback = {
            'layout': layout.tolist(),
            'requirements': asdict(requirements),
            'rating': rating,  # 0-1 scale
            'comments': comments,
            'timestamp': str(datetime.now())
        }
        
        self.feedback_data.append(feedback)
        self.save()
    
    def save(self):
        """Save feedback data"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def get_examples(self) -> List[BaseExample]:
        """Convert feedback to training examples"""
        examples = []
        
        for feedback in self.feedback_data:
            if feedback['rating'] > 0.6:  # Only use positive examples
                layout = np.array(feedback['layout'])
                
                # Reconstruct requirements
                req_dict = feedback['requirements']
                requirements = BaseRequirements(
                    num_colonists=req_dict['num_colonists'],
                    num_bedrooms=req_dict['num_bedrooms'],
                    num_workshops=req_dict['num_workshops'],
                    has_kitchen=req_dict['has_kitchen'],
                    has_hospital=req_dict['has_hospital'],
                    has_recreation=req_dict['has_recreation'],
                    defense_level=req_dict['defense_level'],
                    beauty_preference=req_dict['beauty_preference'],
                    efficiency_preference=req_dict['efficiency_preference'],
                    size_constraint=tuple(req_dict['size_constraint'])
                )
                
                # Use rating to determine quality scores
                quality = {
                    'efficiency': feedback['rating'],
                    'beauty': feedback['rating'],
                    'defense': 0.5,  # Default
                    'connectivity': feedback['rating'],
                    'space_usage': feedback['rating'],
                }
                
                example = BaseExample(
                    layout=layout,
                    requirements=requirements,
                    quality_scores=quality,
                    source="user_feedback",
                    description=feedback.get('comments', '')
                )
                examples.append(example)
                
        return examples


class BaseDataset(Dataset):
    """PyTorch dataset for base examples"""
    
    def __init__(self, examples: List[BaseExample], augment: bool = True):
        self.examples = examples
        self.augment = augment
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        layout, requirements, quality = example.to_tensor()
        
        # Ensure fixed size (128x128) by padding or cropping
        target_size = 128
        h, w = layout.shape
        
        if h != target_size or w != target_size:
            # Create padded/cropped layout
            fixed_layout = torch.zeros(target_size, target_size, dtype=layout.dtype)
            
            # Calculate crop/pad amounts
            h_start = max(0, (h - target_size) // 2)
            w_start = max(0, (w - target_size) // 2)
            h_end = min(h, h_start + target_size)
            w_end = min(w, w_start + target_size)
            
            # Calculate destination coordinates
            dest_h_start = max(0, (target_size - h) // 2)
            dest_w_start = max(0, (target_size - w) // 2)
            dest_h_end = dest_h_start + (h_end - h_start)
            dest_w_end = dest_w_start + (w_end - w_start)
            
            # Copy the data
            fixed_layout[dest_h_start:dest_h_end, dest_w_start:dest_w_end] = \
                layout[h_start:h_end, w_start:w_end]
            
            layout = fixed_layout
        
        if self.augment:
            # Random rotation
            if torch.rand(1) > 0.5:
                layout = torch.rot90(layout, k=torch.randint(1, 4, (1,)).item())
            
            # Random flip
            if torch.rand(1) > 0.5:
                layout = torch.flip(layout, [0])
            if torch.rand(1) > 0.5:
                layout = torch.flip(layout, [1])
        
        # Convert to one-hot encoding  
        num_classes = 256  # Extended number of cell types for mod content
        layout_onehot = torch.nn.functional.one_hot(layout.long(), num_classes)
        layout_onehot = layout_onehot.permute(2, 0, 1).float()
        
        return layout_onehot, requirements, quality


class DatasetCollector:
    """Main dataset collector combining all sources"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        self.prefabs_parser = AlphaPrefabsParser(
            Path("data/AlphaPrefabs/1.5/Defs/LayoutDefs")
        )
        self.feedback_collector = UserFeedbackCollector(
            self.data_dir / "user_feedback.json"
        )
    
    def collect_all_data(self) -> List[BaseExample]:
        """Collect data from all sources"""
        all_examples = []
        
        # Collect from AlphaPrefabs
        logger.info("Collecting from AlphaPrefabs...")
        prefab_examples = self.prefabs_parser.collect_all()
        all_examples.extend(prefab_examples)
        logger.info(f"Collected {len(prefab_examples)} from AlphaPrefabs")
        
        # Collect from user feedback
        logger.info("Collecting from user feedback...")
        feedback_examples = self.feedback_collector.get_examples()
        all_examples.extend(feedback_examples)
        logger.info(f"Collected {len(feedback_examples)} from user feedback")
        
        return all_examples
    
    def save_dataset(self, examples: List[BaseExample], filename: str = "base_dataset.pkl"):
        """Save dataset to file"""
        filepath = self.data_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(examples, f)
        logger.info(f"Saved {len(examples)} examples to {filepath}")
    
    def load_dataset(self, filename: str = "base_dataset.pkl") -> List[BaseExample]:
        """Load dataset from file"""
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                examples = pickle.load(f)
            logger.info(f"Loaded {len(examples)} examples from {filepath}")
            return examples
        return []
    
    def create_data_loaders(self, examples: List[BaseExample], 
                           batch_size: int = 16,
                           train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        # Split data
        n_train = int(len(examples) * train_split)
        train_examples = examples[:n_train]
        val_examples = examples[n_train:]
        
        # Create datasets
        train_dataset = BaseDataset(train_examples, augment=True)
        val_dataset = BaseDataset(val_examples, augment=False)
        
        # Create loaders
        # Set num_workers=0 on Windows to avoid multiprocessing issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Changed from 2 to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Changed from 2 to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
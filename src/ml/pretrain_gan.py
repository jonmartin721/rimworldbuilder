#!/usr/bin/env python3
"""
Pre-train GAN using rule-based generator examples
This gives the GAN a head start by learning from good examples
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
from tqdm import tqdm
import logging

from src.generators.realistic_base_generator import RealisticBaseGenerator
from src.ml.base_gan_model import BaseGAN, BaseRequirements
from src.visualization.realistic_visualizer import RealisticBaseVisualizer

logger = logging.getLogger(__name__)


class GANPretrainer:
    """Pre-trains GAN using high-quality rule-based examples"""
    
    def __init__(self, model_dir: Path = Path("models/base_gan")):
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize GAN
        self.gan = BaseGAN()
        self.gan.generator.to(self.device)
        self.gan.discriminator.to(self.device)
        
        # Data directory
        self.data_dir = Path("data/ml_training")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized pretrainer on {self.device}")
    
    def generate_rule_based_dataset(self, num_examples: int = 500) -> List[Dict]:
        """Generate high-quality examples using rule-based generator"""
        logger.info(f"Generating {num_examples} rule-based examples...")
        
        examples = []
        visualizer = RealisticBaseVisualizer(scale=4)
        
        for i in tqdm(range(num_examples), desc="Generating examples"):
            # Vary parameters for diversity
            num_bedrooms = np.random.randint(3, 12)
            num_colonists = num_bedrooms + np.random.randint(0, 4)
            
            # Create rule-based layout
            generator = RealisticBaseGenerator(128, 128)
            layout = generator.generate_base(
                num_bedrooms=num_bedrooms,
                include_kitchen=True,
                include_workshop=np.random.randint(1, 4),
                include_hospital=np.random.rand() > 0.3,
                include_storage=np.random.rand() > 0.2,
                include_dining=np.random.rand() > 0.4,
                include_recreation=np.random.rand() > 0.5,
                include_prison=np.random.rand() > 0.7
            )
            
            # Create requirements that match the generated base
            requirements = BaseRequirements(
                num_colonists=num_colonists,
                num_bedrooms=num_bedrooms,
                num_workshops=np.random.randint(1, 4),
                has_kitchen=True,
                has_hospital=np.random.rand() > 0.3,
                has_recreation=np.random.rand() > 0.5,
                defense_level=np.random.rand(),
                beauty_preference=np.random.rand(),
                efficiency_preference=0.5 + np.random.rand() * 0.5,
                size_constraint=(128, 128)
            )
            
            # Convert layout to normalized format (0-1 range)
            # The layout is categorical, so we normalize by max category value
            layout_normalized = layout.astype(np.float32) / 10.0  # Assuming max 10 room types
            
            # Quality scores (rule-based are high quality)
            quality_scores = {
                'overall': 0.7 + np.random.rand() * 0.3,  # 0.7-1.0 range
                'efficiency': 0.6 + np.random.rand() * 0.4,
                'beauty': 0.5 + np.random.rand() * 0.5,
                'defense': requirements.defense_level,
                'room_quality': 0.8 + np.random.rand() * 0.2
            }
            
            example = {
                'layout': layout_normalized,
                'requirements': requirements,
                'quality_scores': quality_scores,
                'source': 'rule_based',
                'original_layout': layout  # Keep original for visualization
            }
            examples.append(example)
            
            # Save preview of first few
            if i < 5:
                preview_path = self.data_dir / f"pretrain_example_{i}.png"
                visualizer.visualize(layout, str(preview_path), 
                                   title=f"Rule-based Example {i+1}")
        
        # Save dataset
        dataset_path = self.data_dir / "rule_based_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(examples, f)
        
        logger.info(f"Saved {len(examples)} examples to {dataset_path}")
        return examples
    
    def create_training_batch(self, examples: List[Dict], batch_size: int = 16) -> Tuple:
        """Create a training batch from examples"""
        # Sample random examples
        indices = np.random.choice(len(examples), batch_size, replace=False)
        batch_examples = [examples[i] for i in indices]
        
        # Prepare tensors
        layouts = []
        conditions = []
        quality_labels = []
        
        for ex in batch_examples:
            # Layout (add channel dimension)
            layout = ex['layout']
            if len(layout.shape) == 2:
                layout = np.expand_dims(layout, 0)  # Add channel dimension
            layouts.append(layout)
            
            # Conditions from requirements
            req = ex['requirements']
            condition_vec = np.array([
                req.num_colonists / 20.0,  # Normalize to 0-1
                req.num_bedrooms / 20.0,
                req.num_workshops / 10.0,
                float(req.has_kitchen),
                float(req.has_hospital),
                float(req.has_recreation),
                req.defense_level,
                req.beauty_preference,
                req.efficiency_preference,
                req.size_constraint[0] / 256.0,
                req.size_constraint[1] / 256.0
            ], dtype=np.float32)
            conditions.append(condition_vec)
            
            # Quality label (overall score)
            quality = ex['quality_scores']['overall']
            quality_labels.append(quality)
        
        # Convert to tensors
        layouts = torch.FloatTensor(np.array(layouts))
        conditions = torch.FloatTensor(np.array(conditions))
        quality_labels = torch.FloatTensor(np.array(quality_labels))
        
        return layouts, conditions, quality_labels
    
    def pretrain(self, num_epochs: int = 50, batch_size: int = 16):
        """Pre-train the GAN on rule-based examples"""
        # Generate or load dataset
        dataset_path = self.data_dir / "rule_based_dataset.pkl"
        if dataset_path.exists():
            logger.info("Loading existing rule-based dataset...")
            with open(dataset_path, 'rb') as f:
                examples = pickle.load(f)
            logger.info(f"Loaded {len(examples)} examples")
        else:
            examples = self.generate_rule_based_dataset(500)
        
        # Training loop
        logger.info(f"Starting pre-training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            num_batches = len(examples) // batch_size
            
            for batch_idx in range(num_batches):
                # Get batch
                layouts, conditions, quality_labels = self.create_training_batch(examples, batch_size)
                layouts = layouts.to(self.device)
                conditions = conditions.to(self.device)
                quality_labels = quality_labels.to(self.device)
                
                # Train discriminator
                self.gan.optimizer_d.zero_grad()
                
                # Real samples
                real_validity = self.gan.discriminator(layouts, conditions)
                real_labels = torch.ones_like(real_validity) * 0.9  # Label smoothing
                d_real_loss = self.gan.criterion(real_validity, real_labels)
                
                # Fake samples
                z = torch.randn(batch_size, self.gan.latent_dim).to(self.device)
                fake_layouts = self.gan.generator(z, conditions)
                fake_validity = self.gan.discriminator(fake_layouts.detach(), conditions)
                fake_labels = torch.zeros_like(fake_validity) * 0.1  # Label smoothing
                d_fake_loss = self.gan.criterion(fake_validity, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.gan.optimizer_d.step()
                
                # Train generator
                self.gan.optimizer_g.zero_grad()
                
                z = torch.randn(batch_size, self.gan.latent_dim).to(self.device)
                gen_layouts = self.gan.generator(z, conditions)
                gen_validity = self.gan.discriminator(gen_layouts, conditions)
                gen_labels = torch.ones_like(gen_validity)  # Want to fool discriminator
                
                g_loss = self.gan.criterion(gen_validity, gen_labels)
                g_loss.backward()
                self.gan.optimizer_g.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            # Log progress
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                
                # Generate sample
                self.generate_sample(epoch + 1)
                
                # Save checkpoint
                checkpoint_path = self.model_dir / f"pretrained_epoch_{epoch+1}.pt"
                self.gan.save(checkpoint_path)
        
        # Save final model
        final_path = self.model_dir / "pretrained_model.pt"
        self.gan.save(final_path)
        logger.info(f"Pre-training complete! Model saved to {final_path}")
    
    def generate_sample(self, epoch: int):
        """Generate a sample to visualize progress"""
        self.gan.generator.eval()
        
        with torch.no_grad():
            # Create sample requirements
            requirements = BaseRequirements(
                num_colonists=8,
                num_bedrooms=6,
                num_workshops=2,
                has_kitchen=True,
                has_hospital=True,
                has_recreation=True,
                defense_level=0.5,
                beauty_preference=0.7,
                efficiency_preference=0.8,
                size_constraint=(128, 128)
            )
            
            # Generate
            layout = self.gan.generate(requirements)
            
            # Convert back to categorical
            layout_categorical = (layout * 10).astype(np.int32)
            
            # Visualize
            visualizer = RealisticBaseVisualizer(scale=6)
            output_path = self.model_dir / f"pretrain_sample_epoch_{epoch}.png"
            visualizer.visualize(layout_categorical, str(output_path),
                               title=f"Pre-training Epoch {epoch}")
            
            logger.info(f"Generated sample saved to {output_path}")
        
        self.gan.generator.train()


def main():
    """Run pre-training"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    pretrainer = GANPretrainer()
    
    # Generate rule-based dataset if needed
    dataset_path = Path("data/ml_training/rule_based_dataset.pkl")
    if not dataset_path.exists():
        print("Generating rule-based training data...")
        pretrainer.generate_rule_based_dataset(1000)  # Generate 1000 examples
    
    # Pre-train the GAN
    print("Pre-training GAN on rule-based examples...")
    pretrainer.pretrain(num_epochs=100, batch_size=32)
    
    print("\nPre-training complete! The GAN now has a good starting point.")
    print("You can now continue training with the main training GUI.")


if __name__ == "__main__":
    main()
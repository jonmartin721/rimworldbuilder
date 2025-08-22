"""
Training pipeline for the base generation model with interactive feedback.
Supports training on RTX 5090 with mixed precision and distributed training.
"""

import torch
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json

from src.ml.base_gan_model import BaseGAN, BaseRequirements
from src.ml.dataset_collector import DatasetCollector, BaseExample, UserFeedbackCollector
from src.visualization.realistic_visualizer import RealisticBaseVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveTrainer:
    """Interactive training system with user feedback"""
    
    def __init__(self, model_dir: Path, data_dir: Path, feedback_callback=None):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.feedback_callback = feedback_callback  # Callback for getting user feedback
        
        # Initialize model - Check for RTX 5090 compatibility
        self.device = "cpu"  # Default to CPU
        
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                test_tensor = torch.randn(10, 10).cuda()
                _ = test_tensor + test_tensor  # Simple operation
                self.device = "cuda"
                logger.info("CUDA test successful - using GPU")
            except RuntimeError as e:
                if "no kernel image" in str(e):
                    logger.warning("RTX 5090 detected but PyTorch doesn't have compiled kernels for sm_120 yet")
                    logger.warning("Falling back to CPU mode. Training will be slower.")
                    logger.info("For GPU support, you'll need to build PyTorch from source or wait for official RTX 5090 support")
                else:
                    logger.error(f"CUDA error: {e}")
        
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            # Log GPU info
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Set CUDA settings (using new PyTorch 2.9+ API)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.gan = BaseGAN(device=self.device)
        
        # Initialize data collector
        self.collector = DatasetCollector(data_dir)
        self.feedback_collector = UserFeedbackCollector(data_dir / "user_feedback.json")
        
        # Visualizer for showing results
        self.visualizer = RealisticBaseVisualizer(scale=8)
        
        # Training config
        self.config = {
            'batch_size': 32,  # Increased for RTX 5090's 32GB VRAM
            'epochs': 50,  # Reduced for faster initial testing
            'save_interval': 10,
            'feedback_interval': 5,  # Request feedback every 5 epochs
            'learning_rate_g': 0.0001,
            'learning_rate_d': 0.0004,
            'gradient_clip': 1.0,
        }
        
        # Metrics tracking
        self.metrics = {
            'train_g_loss': [],
            'train_d_loss': [],
            'val_g_loss': [],
            'val_d_loss': [],
            'quality_scores': [],
            'user_ratings': []
        }
    
    def prepare_data(self) -> Tuple:
        """Prepare training and validation data"""
        logger.info("Collecting dataset...")
        
        # Collect all examples
        examples = self.collector.collect_all_data()
        
        if len(examples) == 0:
            logger.warning("No training data found! Generating synthetic data...")
            examples = self.generate_synthetic_data(100)
        
        logger.info(f"Total examples: {len(examples)}")
        
        # Create data loaders
        train_loader, val_loader = self.collector.create_data_loaders(
            examples,
            batch_size=self.config['batch_size']
        )
        
        return train_loader, val_loader
    
    def generate_synthetic_data(self, num_examples: int) -> List[BaseExample]:
        """Generate synthetic training data using rule-based generator"""
        from src.generators.realistic_base_generator import RealisticBaseGenerator
        
        examples = []
        
        for i in range(num_examples):
            # Random requirements
            requirements = BaseRequirements(
                num_colonists=np.random.randint(3, 15),
                num_bedrooms=np.random.randint(3, 12),
                num_workshops=np.random.randint(1, 5),
                has_kitchen=np.random.rand() > 0.2,
                has_hospital=np.random.rand() > 0.5,
                has_recreation=np.random.rand() > 0.4,
                defense_level=np.random.rand(),
                beauty_preference=np.random.rand(),
                efficiency_preference=np.random.rand(),
                size_constraint=(100, 100)
            )
            
            # Generate base
            generator = RealisticBaseGenerator(100, 100)
            layout = generator.generate_base(
                num_bedrooms=requirements.num_bedrooms,
                include_kitchen=requirements.has_kitchen,
                include_workshop=requirements.num_workshops,
                include_hospital=requirements.has_hospital
            )
            
            # Random quality scores
            quality = {
                'efficiency': np.random.rand() * 0.5 + 0.5,
                'beauty': np.random.rand() * 0.5 + 0.3,
                'defense': requirements.defense_level,
                'connectivity': np.random.rand() * 0.5 + 0.5,
                'space_usage': np.random.rand() * 0.5 + 0.4,
            }
            
            example = BaseExample(
                layout=layout,
                requirements=requirements,
                quality_scores=quality,
                source="synthetic",
                description=f"Synthetic base {i+1}"
            )
            examples.append(example)
        
        # Save synthetic data
        self.collector.save_dataset(examples, "synthetic_dataset.pkl")
        
        return examples
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_quality = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (layouts, conditions, quality_labels) in enumerate(pbar):
            layouts = layouts.to(self.device)
            conditions = conditions.to(self.device)
            quality_labels = quality_labels.to(self.device)
            
            # Train step
            metrics = self.gan.train_step(layouts, conditions, quality_labels)
            
            # Update metrics
            epoch_g_loss += metrics['g_loss']
            epoch_d_loss += metrics['d_loss']
            epoch_quality += metrics['quality_score']
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{metrics['g_loss']:.4f}",
                'D': f"{metrics['d_loss']:.4f}",
                'Q': f"{metrics['quality_score']:.3f}"
            })
        
        # Average metrics
        n_batches = len(train_loader)
        return {
            'g_loss': epoch_g_loss / n_batches,
            'd_loss': epoch_d_loss / n_batches,
            'quality': epoch_quality / n_batches
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model"""
        self.gan.generator.eval()
        self.gan.discriminator.eval()
        
        val_g_loss = 0
        val_d_loss = 0
        val_quality = 0
        
        with torch.no_grad():
            for layouts, conditions, quality_labels in val_loader:
                layouts = layouts.to(self.device)
                conditions = conditions.to(self.device)
                quality_labels = quality_labels.to(self.device)
                
                # Generate fake bases
                z = torch.randn(layouts.size(0), self.gan.latent_dim).to(self.device)
                with autocast():
                    fake_bases = self.gan.generator(z, conditions)
                    
                    # Discriminator evaluation
                    real_validity, real_quality = self.gan.discriminator(layouts, conditions)
                    fake_validity, fake_quality = self.gan.discriminator(fake_bases, conditions)
                    
                    # Calculate losses
                    d_real = self.gan.adversarial_loss(real_validity, torch.ones_like(real_validity))
                    d_fake = self.gan.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
                    g_loss = self.gan.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
                    
                    val_d_loss += (d_real + d_fake).item() / 2
                    val_g_loss += g_loss.item()
                    val_quality += fake_quality.mean().item()
        
        n_batches = len(val_loader)
        return {
            'g_loss': val_g_loss / n_batches,
            'd_loss': val_d_loss / n_batches,
            'quality': val_quality / n_batches
        }
    
    def get_user_feedback_via_callback(self, num_samples: int = 3) -> List[float]:
        """Generate samples and get user feedback via callback"""
        logger.info("\n" + "="*60)
        logger.info("REQUESTING USER FEEDBACK")
        logger.info("="*60)
        
        samples_data = []
        
        for i in range(num_samples):
            # Generate random requirements
            requirements = BaseRequirements(
                num_colonists=np.random.randint(5, 10),
                num_bedrooms=np.random.randint(4, 8),
                num_workshops=np.random.randint(1, 3),
                has_kitchen=True,
                has_hospital=np.random.rand() > 0.5,
                has_recreation=np.random.rand() > 0.5,
                defense_level=np.random.rand(),
                beauty_preference=np.random.rand(),
                efficiency_preference=np.random.rand(),
                size_constraint=(128, 128)
            )
            
            # Generate base
            bases = self.gan.generate(requirements, num_samples=1)
            base = bases[0]
            
            # Save visualization
            output_path = self.model_dir / f"feedback_sample_{i+1}.png"
            self.visualizer.visualize(
                base,
                str(output_path),
                title=f"Sample {i+1}: {requirements.num_colonists} colonists"
            )
            
            samples_data.append({
                'image_path': str(output_path),
                'description': f"{requirements.num_colonists} colonists, {requirements.num_bedrooms} bedrooms",
                'base': base,
                'requirements': requirements
            })
        
        # Call the feedback callback and wait for response
        if self.feedback_callback:
            ratings = self.feedback_callback(samples_data)
            
            # Save feedback for each rating
            for i, rating_data in enumerate(ratings):
                if i < len(samples_data):
                    self.feedback_collector.add_feedback(
                        samples_data[i]['base'],
                        samples_data[i]['requirements'],
                        rating_data['rating'],
                        rating_data.get('comments', '')
                    )
            
            return [r['rating'] for r in ratings]
        
        return []
    
    def get_user_feedback(self, num_samples: int = 3) -> List[float]:
        """Generate samples and get user feedback"""
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE FEEDBACK SESSION")
        logger.info("="*60)
        
        ratings = []
        
        for i in range(num_samples):
            # Generate random requirements
            requirements = BaseRequirements(
                num_colonists=np.random.randint(5, 10),
                num_bedrooms=np.random.randint(4, 8),
                num_workshops=np.random.randint(1, 3),
                has_kitchen=True,
                has_hospital=np.random.rand() > 0.5,
                has_recreation=np.random.rand() > 0.5,
                defense_level=np.random.rand(),
                beauty_preference=np.random.rand(),
                efficiency_preference=np.random.rand(),
                size_constraint=(100, 100)
            )
            
            # Generate base
            bases = self.gan.generate(requirements, num_samples=1)
            base = bases[0]
            
            # Save visualization
            output_path = self.model_dir / f"feedback_sample_{i+1}.png"
            self.visualizer.visualize(
                base,
                str(output_path),
                title=f"Sample {i+1}: {requirements.num_colonists} colonists"
            )
            
            # Show ASCII preview
            print(f"\nSample {i+1}/{num_samples}")
            print(f"Requirements: {requirements.num_colonists} colonists, "
                  f"{requirements.num_bedrooms} bedrooms")
            print("-" * 40)
            
            # Show center portion
            center_y, center_x = base.shape[0] // 2, base.shape[1] // 2
            preview = base[center_y-10:center_y+10, center_x-15:center_x+15]
            ascii_preview = self.visualizer.visualize_ascii(preview)
            print(ascii_preview)
            
            print(f"\nVisualization saved to: {output_path}")
            
            # Get rating
            while True:
                try:
                    rating = float(input("Rate this base (0-10): ")) / 10.0
                    if 0 <= rating <= 1:
                        break
                    print("Please enter a value between 0 and 10")
                except ValueError:
                    print("Please enter a valid number")
            
            comments = input("Comments (optional): ")
            
            # Save feedback
            self.feedback_collector.add_feedback(
                base, requirements, rating, comments
            )
            ratings.append(rating)
            
            print(f"Thank you! Rating: {rating:.1f}/1.0")
        
        return ratings
    
    def train(self, epochs: Optional[int] = None):
        """Main training loop"""
        if epochs:
            self.config['epochs'] = epochs
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        logger.info("\n" + "="*60)
        logger.info("STARTING TRAINING")
        logger.info(f"Config: {self.config}")
        logger.info("="*60)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            logger.info(f"\nEpoch {epoch}/{self.config['epochs']}")
            logger.info(f"Train - G: {train_metrics['g_loss']:.4f}, "
                       f"D: {train_metrics['d_loss']:.4f}, "
                       f"Q: {train_metrics['quality']:.3f}")
            logger.info(f"Val   - G: {val_metrics['g_loss']:.4f}, "
                       f"D: {val_metrics['d_loss']:.4f}, "
                       f"Q: {val_metrics['quality']:.3f}")
            
            # Save metrics
            self.metrics['train_g_loss'].append(train_metrics['g_loss'])
            self.metrics['train_d_loss'].append(train_metrics['d_loss'])
            self.metrics['val_g_loss'].append(val_metrics['g_loss'])
            self.metrics['val_d_loss'].append(val_metrics['d_loss'])
            self.metrics['quality_scores'].append(val_metrics['quality'])
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch}.pt"
                self.gan.save(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_metrics['g_loss'] < best_val_loss:
                best_val_loss = val_metrics['g_loss']
                best_path = self.model_dir / "best_model.pt"
                self.gan.save(best_path)
                logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Get user feedback  
            if epoch % self.config['feedback_interval'] == 0:
                logger.info(f"=== FEEDBACK TIME: Epoch {epoch} ===")
                if self.feedback_callback:
                    logger.info("Requesting user feedback via GUI...")
                    # Use callback for GUI mode
                    ratings = self.get_user_feedback_via_callback()
                else:
                    # Skip feedback if no callback provided
                    logger.info("Skipping feedback (no callback provided)")
                    ratings = []
                
                if ratings:
                    avg_rating = np.mean(ratings)
                    self.metrics['user_ratings'].append(avg_rating)
                    logger.info(f"Average user rating: {avg_rating:.2f}/1.0")
                    
                    # Reload data with new feedback
                    if avg_rating > 0.6:
                        logger.info("Good ratings! Reloading data with feedback...")
                        train_loader, val_loader = self.prepare_data()
        
        # Save final model
        final_path = self.model_dir / "final_model.pt"
        self.gan.save(final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")
        
        # Save metrics
        self.save_metrics()
        self.plot_metrics()
    
    def save_metrics(self):
        """Save training metrics"""
        metrics_path = self.model_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def plot_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Generator loss
        axes[0, 0].plot(self.metrics['train_g_loss'], label='Train')
        axes[0, 0].plot(self.metrics['val_g_loss'], label='Val')
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Discriminator loss
        axes[0, 1].plot(self.metrics['train_d_loss'], label='Train')
        axes[0, 1].plot(self.metrics['val_d_loss'], label='Val')
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Quality scores
        axes[1, 0].plot(self.metrics['quality_scores'])
        axes[1, 0].set_title('Quality Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        
        # User ratings
        if self.metrics['user_ratings']:
            epochs = list(range(self.config['feedback_interval'], 
                               len(self.metrics['user_ratings']) * self.config['feedback_interval'] + 1,
                               self.config['feedback_interval']))
            axes[1, 1].plot(epochs, self.metrics['user_ratings'], 'o-')
            axes[1, 1].set_title('User Ratings')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Rating')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.model_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Saved plots to {plot_path}")
        plt.close()


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RimWorld base generator")
    parser.add_argument('--model-dir', type=str, default='models/base_gan',
                       help='Directory to save models')
    parser.add_argument('--data-dir', type=str, default='data/ml_training',
                       help='Directory for training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = InteractiveTrainer(
        model_dir=Path(args.model_dir),
        data_dir=Path(args.data_dir)
    )
    
    # Update config
    trainer.config['batch_size'] = args.batch_size
    
    # Resume if specified
    if args.resume:
        trainer.gan.load(Path(args.resume))
        logger.info(f"Resumed from {args.resume}")
    
    # Start training
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
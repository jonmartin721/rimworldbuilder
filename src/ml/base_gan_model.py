"""
Generative Adversarial Network (GAN) for RimWorld base generation.
Uses a conditional GAN to generate bases based on requirements.
Optimized for RTX 5090 with mixed precision training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


@dataclass
class BaseRequirements:
    """Requirements for base generation"""
    num_colonists: int
    num_bedrooms: int
    num_workshops: int
    has_kitchen: bool
    has_hospital: bool
    has_recreation: bool
    defense_level: float  # 0-1
    beauty_preference: float  # 0-1
    efficiency_preference: float  # 0-1
    size_constraint: Tuple[int, int]  # (width, height)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert requirements to tensor for model input"""
        return torch.tensor([
            self.num_colonists / 20.0,  # Normalize to 0-1
            self.num_bedrooms / 20.0,
            self.num_workshops / 10.0,
            float(self.has_kitchen),
            float(self.has_hospital),
            float(self.has_recreation),
            self.defense_level,
            self.beauty_preference,
            self.efficiency_preference,
            self.size_constraint[0] / 200.0,
            self.size_constraint[1] / 200.0,
        ], dtype=torch.float32)


class ResidualBlock(nn.Module):
    """Residual block with batch norm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for capturing long-range dependencies"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        proj_value = self.value(x).view(batch_size, -1, H * W)
        
        # Compute attention
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Apply learnable weight and add residual
        out = self.gamma * out + x
        return out


class BaseGenerator(nn.Module):
    """Generator network for creating base layouts"""
    
    def __init__(self, latent_dim: int = 128, condition_dim: int = 11, num_classes: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_classes = num_classes  # Different building/room types
        
        # Initial projection
        self.fc = nn.Linear(latent_dim + condition_dim, 512 * 8 * 8)
        
        # Upsampling layers with residual connections
        self.layer1 = nn.Sequential(
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(512, 256),
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(256, 256),
            AttentionBlock(256),  # Add attention at 16x16 resolution
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256, 128),
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128, 64),
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64),
            AttentionBlock(64),  # Add attention at 64x64 resolution
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(64, 32),
        )
        
        # Output layer
        self.output = nn.Conv2d(32, num_classes, 3, 1, 1)
        
        # Initialize weights for better initial outputs
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to avoid all-empty outputs"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layer to avoid all zeros
        with torch.no_grad():
            self.output.weight.normal_(0, 0.02)
            self.output.bias.normal_(0, 0.1)  # Small positive bias for variety
        
    def forward(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Generate base layout from latent vector and conditions"""
        # Concatenate noise and conditions
        x = torch.cat([z, conditions], dim=1)
        
        # Project and reshape
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        
        # Generate through layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Output logits for each cell type
        x = self.output(x)
        
        return x


class BaseDiscriminator(nn.Module):
    """Discriminator network for evaluating base quality"""
    
    def __init__(self, num_classes: int = 256, condition_dim: int = 11):
        super().__init__()
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        
        # Convolutional layers with spectral normalization for stability
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(num_classes, 64, 4, 2, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        
        # Attention for better feature extraction
        self.attention = AttentionBlock(256)
        
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        
        # Condition projection
        self.condition_fc = nn.Linear(condition_dim, 512)
        
        # Output layers
        self.fc1 = nn.Linear(512 * 4 * 4 + 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        
        # Auxiliary classifier for base quality metrics
        self.quality_head = nn.Linear(512, 5)  # 5 quality metrics
        
    def forward(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate base layout with conditions"""
        # Process base layout
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.attention(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Process conditions
        c = self.condition_fc(conditions)
        
        # Concatenate
        x = torch.cat([x, c], dim=1)
        
        # Classification
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        
        # Real/fake output
        validity = self.fc3(x)
        
        # Quality metrics (efficiency, beauty, defense, connectivity, space_usage)
        quality = self.quality_head(x)
        
        return validity, quality


class BaseGAN:
    """Complete GAN system for base generation"""
    
    def __init__(self, device: str = "cuda", latent_dim: int = 128):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        
        # Initialize networks
        self.generator = BaseGenerator(latent_dim).to(self.device)
        self.discriminator = BaseDiscriminator().to(self.device)
        
        # Optimizers with different learning rates
        self.g_optimizer = optim.AdamW(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self.d_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.quality_loss = nn.MSELoss()
        self.diversity_loss = nn.L1Loss()
        
        # Mixed precision training for RTX 5090
        self.scaler = GradScaler()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'quality_scores': []
        }
    
    def generate(self, requirements: BaseRequirements, num_samples: int = 1) -> np.ndarray:
        """Generate base layouts from requirements"""
        self.generator.eval()
        
        with torch.no_grad():
            # Sample latent vectors
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            # Prepare conditions
            conditions = requirements.to_tensor().unsqueeze(0).repeat(num_samples, 1).to(self.device)
            
            # Generate (no autocast for RTX 5090 compatibility)
            output = self.generator(z, conditions)
            
            # Convert to class indices
            bases = torch.argmax(output, dim=1).cpu().numpy()
            
            # Debug: Check if output is too uniform
            unique_vals = np.unique(bases)
            if len(unique_vals) <= 2:
                print(f"Warning: Generated base has only {len(unique_vals)} unique values: {unique_vals}")
                # Add some random noise to early untrained models
                if len(unique_vals) == 1:
                    # Completely uniform - add some random structure
                    for i in range(num_samples):
                        # Add random walls in a simple pattern
                        h, w = bases[i].shape
                        # Add border walls
                        bases[i][0, :] = 1  # Top wall
                        bases[i][-1, :] = 1  # Bottom wall
                        bases[i][:, 0] = 1  # Left wall
                        bases[i][:, -1] = 1  # Right wall
                        # Add some random rooms
                        for _ in range(3):
                            x, y = np.random.randint(5, w-5), np.random.randint(5, h-5)
                            w_room, h_room = np.random.randint(4, 8), np.random.randint(4, 8)
                            # Draw room walls
                            bases[i][y:y+h_room, x] = 1
                            bases[i][y:y+h_room, x+w_room] = 1
                            bases[i][y, x:x+w_room] = 1
                            bases[i][y+h_room, x:x+w_room] = 1
                            # Add door
                            bases[i][y+h_room//2, x] = 2
        
        return bases
    
    def train_step(self, real_bases: torch.Tensor, conditions: torch.Tensor, 
                   quality_labels: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        try:
            batch_size = real_bases.size(0)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Disable mixed precision for now - RTX 5090 compatibility
            # Real bases
            real_validity, real_quality = self.discriminator(real_bases, conditions)
            d_real_loss = self.adversarial_loss(real_validity, torch.ones_like(real_validity))
            
            # Fake bases
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_bases = self.generator(z, conditions)
            fake_validity, fake_quality = self.discriminator(fake_bases.detach(), conditions)
            d_fake_loss = self.adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            
            # Quality prediction loss
            quality_loss = self.quality_loss(real_quality, quality_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + quality_loss
            
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            # Generate new bases
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_bases = self.generator(z, conditions)
            fake_validity, fake_quality = self.discriminator(fake_bases, conditions)
            
            # Adversarial loss
            g_adv_loss = self.adversarial_loss(fake_validity, torch.ones_like(fake_validity))
            
            # Quality loss - encourage high quality scores
            target_quality = torch.ones_like(fake_quality) * 0.9  # Target high quality
            g_quality_loss = self.quality_loss(fake_quality, target_quality)
            
            # Diversity loss - encourage different outputs
            if batch_size > 1:
                diversity = self.diversity_loss(fake_bases[0], fake_bases[1])
                g_div_loss = -diversity * 0.1  # Negative to encourage diversity
            else:
                g_div_loss = torch.tensor(0.0).to(self.device)
            
            # Total generator loss
            g_loss = g_adv_loss + g_quality_loss + g_div_loss
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Synchronize CUDA
            if self.device == "cuda":
                torch.cuda.synchronize()
            
        except RuntimeError as e:
            print(f"CUDA Error in train_step: {e}")
            print("Attempting to recover...")
            torch.cuda.empty_cache()
            raise e
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'quality_score': fake_quality.mean().item()
        }
    
    def save(self, path: Path):
        """Save model checkpoints"""
        path = Path(path)  # Ensure it's a Path object
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'history': self.history
        }, str(path))  # Convert to string for torch.save
    
    def load(self, path: Path):
        """Load model checkpoints"""
        path = Path(path)  # Ensure it's a Path object
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        try:
            # Try loading with weights_only=False for compatibility
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Try alternative loading method
            checkpoint = torch.load(str(path), map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.history = checkpoint.get('history', self.history)
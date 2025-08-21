#!/usr/bin/env python3
"""
ML Training GUI for RimWorld Base Generator
Manages training, monitoring, and testing of the GAN model on RTX 5090
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, simpledialog
import threading
import queue
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import GPUtil
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.base_gan_model import BaseRequirements
from src.ml.train_base_model import InteractiveTrainer
from src.ml.dataset_collector import DatasetCollector
from src.ml.side_by_side_feedback import show_side_by_side_feedback
from src.visualization.realistic_visualizer import RealisticBaseVisualizer


class GPUMonitor:
    """Monitor GPU usage and stats"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if not self.gpu_available:
            return {
                'available': False,
                'name': 'No GPU',
                'memory_used': 0,
                'memory_total': 0,
                'memory_percent': 0,
                'temperature': 0,
                'utilization': 0
            }
        
        try:
            if self.gpu:
                self.gpu = GPUtil.getGPUs()[0]  # Refresh
                # GPUtil returns memory in MB, convert to GB
                memory_used_gb = self.gpu.memoryUsed / 1024.0  # Convert MB to GB
                memory_total_gb = self.gpu.memoryTotal / 1024.0  # Convert MB to GB
                
                
                return {
                    'available': True,
                    'name': self.gpu.name,
                    'memory_used': memory_used_gb,
                    'memory_total': memory_total_gb,
                    'memory_percent': (self.gpu.memoryUsed / self.gpu.memoryTotal * 100) if self.gpu.memoryTotal > 0 else 0,
                    'temperature': self.gpu.temperature,
                    'utilization': self.gpu.load * 100
                }
            else:
                # Fallback to PyTorch stats
                memory_used_gb = torch.cuda.memory_allocated(0) / (1024**3)  # Bytes to GB
                memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Bytes to GB
                
                print(f"DEBUG: PyTorch - Memory Used: {memory_used_gb:.2f} GB")
                print(f"DEBUG: PyTorch - Memory Total: {memory_total_gb:.2f} GB")
                
                return {
                    'available': True,
                    'name': torch.cuda.get_device_name(0),
                    'memory_used': memory_used_gb,
                    'memory_total': memory_total_gb,
                    'memory_percent': (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100),
                    'temperature': 0,
                    'utilization': 0
                }
        except Exception as e:
            print(f"ERROR getting GPU stats: {e}")
            import traceback
            traceback.print_exc()
            return {
                'available': False,
                'name': 'Error reading GPU',
                'memory_used': 0,
                'memory_total': 0,
                'memory_percent': 0,
                'temperature': 0,
                'utilization': 0
            }


class TrainingThread(threading.Thread):
    """Background thread for training"""
    
    def __init__(self, trainer, epochs, message_queue, control_queue):
        super().__init__(daemon=True)
        self.trainer = trainer
        self.epochs = epochs
        self.message_queue = message_queue
        self.control_queue = control_queue
        self.running = True
        self.paused = False
        self.feedback_event = threading.Event()
        self.feedback_response = None
    
    def feedback_callback(self, samples_data):
        """Callback for feedback requests from trainer"""
        # Send feedback request to GUI
        self.message_queue.put(('feedback_request', samples_data))
        
        # Wait for response
        self.feedback_event.clear()
        self.feedback_event.wait(timeout=300)  # 5 minute timeout
        
        if self.feedback_response is not None:
            response = self.feedback_response
            self.feedback_response = None
            return response
        else:
            # Timeout or no response - return default ratings
            return [{'rating': 0.5, 'comments': 'No feedback provided'} for _ in samples_data]
    
    def run(self):
        """Run training in background"""
        try:
            self.message_queue.put(('status', 'Training started'))
            
            # Set the feedback callback
            self.trainer.feedback_callback = self.feedback_callback
            
            # Override trainer methods to send updates
            original_train_epoch = self.trainer.train_epoch
            
            def train_epoch_with_updates(train_loader, epoch):
                if not self.running:
                    return {'g_loss': 0, 'd_loss': 0, 'quality': 0}
                
                while self.paused and self.running:
                    time.sleep(0.1)
                
                self.message_queue.put(('epoch', epoch))
                result = original_train_epoch(train_loader, epoch)
                self.message_queue.put(('metrics', result))
                
                # Generate preview samples EVERY epoch for visual feedback
                self.message_queue.put(('generate_previews', epoch))
                
                return result
            
            self.trainer.train_epoch = train_epoch_with_updates
            
            # Start training
            self.trainer.train(epochs=self.epochs)
            
            if self.running:
                self.message_queue.put(('complete', 'Training completed successfully'))
        except Exception as e:
            self.message_queue.put(('error', str(e)))
    
    def pause(self):
        """Pause training"""
        self.paused = True
        self.message_queue.put(('status', 'Training paused'))
    
    def resume(self):
        """Resume training"""
        self.paused = False
        self.message_queue.put(('status', 'Training resumed'))
    
    def stop(self):
        """Stop training"""
        self.running = False
        self.message_queue.put(('status', 'Training stopped'))


class MLTrainingGUI:
    """Main GUI for ML training management"""
    
    def __init__(self):
        print("="*60)
        print("RimWorld ML Training GUI Starting...")
        print("="*60)
        
        self.root = tk.Tk()
        self.root.title("RimWorld Base Generator - ML Training Manager")
        self.root.geometry("1400x900")
        
        # Set icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except Exception:
            pass
        
        # Initialize components
        print("Initializing GPU monitor...")
        try:
            self.gpu_monitor = GPUMonitor()
            stats = self.gpu_monitor.get_stats()
            print(f"GPU Detection: {stats['name']}")
            print(f"GPU Memory: {stats['memory_total']:.2f} GB")
        except Exception as e:
            print(f"ERROR initializing GPU monitor: {e}")
            import traceback
            traceback.print_exc()
            self.gpu_monitor = None
            
        self.trainer = None
        self.model = None  # Initialize model attribute
        self.training_thread = None
        self.message_queue = queue.Queue()
        self.control_queue = queue.Queue()
        
        # Model paths
        self.model_dir = Path("models/base_gan")
        self.data_dir = Path("data/ml_training")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 100
        self.metrics_history = {
            'epochs': [],
            'g_loss': [],
            'd_loss': [],
            'quality': []
        }
        
        # Setup GUI
        self.setup_ui()
        
        # Start update loop
        self.update_loop()
        
        # Load existing datasets on startup (after UI is ready)
        self.root.after(100, self.load_initial_datasets)
        
    def load_initial_datasets(self):
        """Load existing datasets on startup"""
        dataset_summary = []
        
        try:
            # Check for existing prefabs dataset
            prefabs_path = self.data_dir / "prefabs_dataset.pkl"
            if prefabs_path.exists():
                import pickle
                with open(prefabs_path, 'rb') as f:
                    examples = pickle.load(f)
                self.log(f"Loaded existing prefabs dataset: {len(examples)} examples")
                dataset_summary.append(f"Prefabs: {len(examples)}")
            else:
                # Auto-collect prefabs if they don't exist
                self.log("No prefabs dataset found. Collecting from AlphaPrefabs...")
                self.collect_prefabs()
                dataset_summary.append("Prefabs: collecting...")
            
            # Check for Reddit scraped data
            reddit_dataset_path = self.data_dir / "reddit_dataset.pkl"
            reddit_dir = Path("data/reddit_bases")
            
            if reddit_dataset_path.exists():
                # Load existing processed Reddit dataset
                import pickle
                with open(reddit_dataset_path, 'rb') as f:
                    reddit_examples = pickle.load(f)
                self.log(f"Loaded existing Reddit dataset: {len(reddit_examples)} examples")
                dataset_summary.append(f"Reddit: {len(reddit_examples)}")
            elif reddit_dir.exists():
                # Process raw Reddit images into dataset
                images = list(reddit_dir.glob("images/*.png")) + list(reddit_dir.glob("images/*.jpg"))
                if images:
                    self.log(f"Found {len(images)} Reddit images. Processing into dataset...")
                    dataset_summary.append(f"Reddit: processing {len(images)} images...")
                    
                    # Convert ALL Reddit images to training format
                    self.process_reddit_images(images)  # Process all images
            
            # Update dataset info with all sources
            if dataset_summary:
                self.dataset_info.set(" | ".join(dataset_summary))
            else:
                self.dataset_info.set("No datasets loaded")
                
        except Exception as e:
            self.log(f"Error loading initial datasets: {e}")
    
    def process_reddit_images(self, image_paths):
        """Process Reddit images into training dataset format"""
        try:
            from PIL import Image
            import numpy as np
            import pickle
            import json
                
            examples = []
            processed = 0
            
            for img_path in image_paths:
                try:
                    # Load image
                    img = Image.open(img_path)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize to standard size for training (128x128 for consistency)
                    img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array (normalized 0-1)
                    img_array = np.array(img_resized) / 255.0
                    
                    # Load metadata if available
                    metadata_path = Path("data/reddit_bases/metadata") / f"{img_path.stem.split('_')[1]}.json"
                    title = img_path.stem
                    score = 0
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            title = metadata.get('title', title)
                            score = metadata.get('score', 0)
                    
                    # Create training example as simple dict (pickleable)
                    example = {
                        'layout': img_array,
                        'source': f"reddit:{img_path.name}",
                        'quality_scores': {'reddit_score': score}
                    }
                    examples.append(example)
                    processed += 1
                    
                    # Close image
                    img.close()
                    
                except Exception as e:
                    self.log(f"Error processing {img_path.name}: {e}")
            
            if examples:
                # Save as pickle dataset
                dataset_path = self.data_dir / "reddit_dataset.pkl"
                with open(dataset_path, 'wb') as f:
                    pickle.dump(examples, f)
                
                self.log(f"Created Reddit dataset: {len(examples)} examples saved to reddit_dataset.pkl")
                
                # Update dataset info
                self.dataset_info.set(f"Reddit dataset created: {len(examples)} examples")
                
        except Exception as e:
            self.log(f"Error creating Reddit dataset: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.setup_training_tab()
        self.setup_monitoring_tab()
        self.setup_testing_tab()
        self.setup_dataset_tab()
        
    def setup_training_tab(self):
        """Setup training control tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Training Control")
        
        # Top frame - GPU Status
        gpu_frame = ttk.LabelFrame(tab, text="GPU Status (RTX 5090)", padding=10)
        gpu_frame.pack(fill='x', padx=5, pady=5)
        
        self.gpu_status_text = tk.StringVar(value="Checking GPU...")
        self.gpu_status_label = ttk.Label(gpu_frame, textvariable=self.gpu_status_text, font=('Consolas', 10))
        self.gpu_status_label.pack()
        
        # Training Configuration
        config_frame = ttk.LabelFrame(tab, text="Training Configuration", padding=10)
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Configuration grid
        ttk.Label(config_frame, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5)
        self.epochs_var = tk.IntVar(value=50)  # Start with 50 for testing, increase to 200+ for better results
        ttk.Spinbox(config_frame, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(config_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(config_frame, text="Learning Rate (G):").grid(row=1, column=0, sticky='w', padx=5)
        self.lr_g_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(config_frame, textvariable=self.lr_g_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(config_frame, text="Learning Rate (D):").grid(row=1, column=2, sticky='w', padx=5)
        self.lr_d_var = tk.DoubleVar(value=0.0004)
        ttk.Entry(config_frame, textvariable=self.lr_d_var, width=10).grid(row=1, column=3, padx=5)
        
        ttk.Label(config_frame, text="Feedback Interval:").grid(row=2, column=0, sticky='w', padx=5)
        self.feedback_var = tk.IntVar(value=10)  # Changed default to 10
        ttk.Spinbox(config_frame, from_=5, to=100, textvariable=self.feedback_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(config_frame, text="Save Interval:").grid(row=2, column=2, sticky='w', padx=5)
        self.save_var = tk.IntVar(value=10)
        ttk.Spinbox(config_frame, from_=1, to=50, textvariable=self.save_var, width=10).grid(row=2, column=3, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training, style='Accent.TButton')
        self.start_btn.pack(side='left', padx=5)
        
        self.pause_btn = ttk.Button(control_frame, text="Pause", command=self.pause_training, state='disabled')
        self.pause_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Load Checkpoint", command=self.load_checkpoint).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Model", command=self.save_model).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Test Preview", command=self.test_preview_generation).pack(side='left', padx=5)
        
        # Progress
        progress_frame = ttk.LabelFrame(tab, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_text = tk.StringVar(value="Not training")
        ttk.Label(progress_frame, textvariable=self.progress_text).pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(pady=5)
        
        # Visual Progress Display
        visual_frame = ttk.LabelFrame(tab, text="Live Training Preview", padding=10)
        visual_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create preview panels
        preview_container = ttk.Frame(visual_frame)
        preview_container.pack(fill='both', expand=True)
        
        # Best sample preview
        best_frame = ttk.LabelFrame(preview_container, text="Best Sample (Current Epoch)")
        best_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.best_image_label = ttk.Label(best_frame, text="No preview yet")
        self.best_image_label.pack()
        self.best_score_label = ttk.Label(best_frame, text="Score: N/A")
        self.best_score_label.pack()
        
        # Worst sample preview
        worst_frame = ttk.LabelFrame(preview_container, text="Worst Sample (Current Epoch)")
        worst_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.worst_image_label = ttk.Label(worst_frame, text="No preview yet")
        self.worst_image_label.pack()
        self.worst_score_label = ttk.Label(worst_frame, text="Score: N/A")
        self.worst_score_label.pack()
        
        # Latest sample preview
        latest_frame = ttk.LabelFrame(preview_container, text="Latest Generated")
        latest_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        self.latest_image_label = ttk.Label(latest_frame, text="No preview yet")
        self.latest_image_label.pack()
        self.latest_info_label = ttk.Label(latest_frame, text="Epoch: N/A")
        self.latest_info_label.pack()
        
        # Configure grid weights
        preview_container.grid_columnconfigure(0, weight=1)
        preview_container.grid_columnconfigure(1, weight=1)
        preview_container.grid_columnconfigure(2, weight=1)
        
        # Training log (smaller now)
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=False, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, width=80, font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)
        
    def setup_monitoring_tab(self):
        """Setup monitoring tab with graphs"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Monitoring")
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Training Metrics')
        
        # Generator loss
        self.axes[0, 0].set_title('Generator Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True)
        
        # Discriminator loss
        self.axes[0, 1].set_title('Discriminator Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True)
        
        # Quality score
        self.axes[1, 0].set_title('Quality Score')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Score')
        self.axes[1, 0].grid(True)
        
        # GPU usage
        self.axes[1, 1].set_title('GPU Memory Usage')
        self.axes[1, 1].set_xlabel('Time')
        self.axes[1, 1].set_ylabel('Memory (GB)')
        self.axes[1, 1].grid(True)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Refresh Plots", command=self.update_plots).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Plots", command=self.save_plots).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear History", command=self.clear_history).pack(side='left', padx=5)
        
    def setup_testing_tab(self):
        """Setup testing tab for generating bases"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Test Generation")
        
        # Requirements frame
        req_frame = ttk.LabelFrame(tab, text="Base Requirements", padding=10)
        req_frame.pack(fill='x', padx=5, pady=5)
        
        # Requirements grid
        ttk.Label(req_frame, text="Colonists:").grid(row=0, column=0, sticky='w', padx=5)
        self.test_colonists = tk.IntVar(value=8)
        ttk.Spinbox(req_frame, from_=1, to=20, textvariable=self.test_colonists, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(req_frame, text="Bedrooms:").grid(row=0, column=2, sticky='w', padx=5)
        self.test_bedrooms = tk.IntVar(value=8)
        ttk.Spinbox(req_frame, from_=1, to=20, textvariable=self.test_bedrooms, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(req_frame, text="Workshops:").grid(row=0, column=4, sticky='w', padx=5)
        self.test_workshops = tk.IntVar(value=3)
        ttk.Spinbox(req_frame, from_=0, to=10, textvariable=self.test_workshops, width=10).grid(row=0, column=5, padx=5)
        
        # Checkboxes
        self.test_kitchen = tk.BooleanVar(value=True)
        ttk.Checkbutton(req_frame, text="Kitchen", variable=self.test_kitchen).grid(row=1, column=0, padx=5)
        
        self.test_hospital = tk.BooleanVar(value=True)
        ttk.Checkbutton(req_frame, text="Hospital", variable=self.test_hospital).grid(row=1, column=1, padx=5)
        
        self.test_recreation = tk.BooleanVar(value=True)
        ttk.Checkbutton(req_frame, text="Recreation", variable=self.test_recreation).grid(row=1, column=2, padx=5)
        
        # Sliders
        ttk.Label(req_frame, text="Defense Level:").grid(row=2, column=0, sticky='w', padx=5)
        self.test_defense = tk.DoubleVar(value=0.5)
        ttk.Scale(req_frame, from_=0, to=1, variable=self.test_defense, orient='horizontal', length=200).grid(row=2, column=1, columnspan=2, padx=5)
        
        ttk.Label(req_frame, text="Beauty:").grid(row=2, column=3, sticky='w', padx=5)
        self.test_beauty = tk.DoubleVar(value=0.7)
        ttk.Scale(req_frame, from_=0, to=1, variable=self.test_beauty, orient='horizontal', length=200).grid(row=2, column=4, columnspan=2, padx=5)
        
        # Generate button
        gen_frame = ttk.Frame(tab)
        gen_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(gen_frame, text="Generate Base (AI)", command=self.generate_test_base, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(gen_frame, text="Generate Multiple (5)", command=lambda: self.generate_test_base(5)).pack(side='left', padx=5)
        ttk.Button(gen_frame, text="Generate Rule-Based", command=self.generate_rule_based).pack(side='left', padx=5)
        ttk.Button(gen_frame, text="Test Side-by-Side", command=self.test_side_by_side_feedback).pack(side='left', padx=5)
        ttk.Button(gen_frame, text="Save Generated", command=self.save_generated).pack(side='left', padx=5)
        
        # Image display
        self.image_frame = ttk.LabelFrame(tab, text="Generated Base", padding=10)
        self.image_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.image_label = ttk.Label(self.image_frame, text="No base generated yet")
        self.image_label.pack()
        
        # Rating frame
        rating_frame = ttk.Frame(tab)
        rating_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(rating_frame, text="Rate this base:").pack(side='left', padx=5)
        self.rating_var = tk.IntVar(value=5)
        for i in range(1, 11):
            ttk.Radiobutton(rating_frame, text=str(i), variable=self.rating_var, value=i).pack(side='left')
        
        ttk.Button(rating_frame, text="Submit Rating", command=self.submit_rating).pack(side='left', padx=10)
        
    def setup_dataset_tab(self):
        """Setup dataset management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Dataset")
        
        # Dataset info
        info_frame = ttk.LabelFrame(tab, text="Dataset Information", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.dataset_info = tk.StringVar(value="No dataset loaded")
        ttk.Label(info_frame, textvariable=self.dataset_info, font=('Consolas', 10)).pack()
        
        # Dataset controls
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="Collect from AlphaPrefabs", command=self.collect_prefabs).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Generate Synthetic Data", command=self.generate_synthetic).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Scrape Reddit (External)", command=self.launch_reddit_scraper).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Load Dataset", command=self.load_dataset).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Dataset", command=self.export_dataset).pack(side='left', padx=5)
        
        # Dataset preview
        preview_frame = ttk.LabelFrame(tab, text="Dataset Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.dataset_text = scrolledtext.ScrolledText(preview_frame, height=20, width=80, font=('Consolas', 9))
        self.dataset_text.pack(fill='both', expand=True)
        
    def update_gpu_status(self):
        """Update GPU status display"""
        if not self.gpu_monitor:
            self.gpu_status_text.set("GPU monitor not available")
            return
            
        try:
            stats = self.gpu_monitor.get_stats()
            
            if stats['available']:
                status = f"GPU: {stats['name']}\n"
                status += f"Memory: {stats['memory_used']:.1f}/{stats['memory_total']:.1f} GB ({stats['memory_percent']:.1f}%)\n"
                status += f"Utilization: {stats['utilization']:.1f}%"
                if stats['temperature'] > 0:
                    status += f" | Temp: {stats['temperature']}°C"
            else:
                status = "No GPU available - Training will use CPU (slow)"
            
            self.gpu_status_text.set(status)
        except Exception as e:
            print(f"ERROR updating GPU status: {e}")
            self.gpu_status_text.set(f"GPU status error: {str(e)}")
        
    def start_training(self):
        """Start training process"""
        if self.is_training:
            print("WARNING: Training already in progress")
            self.log("WARNING: Training already in progress")
            return
        
        try:
            # Initialize trainer if needed
            if self.trainer is None:
                self.log("Initializing trainer...")
                self.trainer = InteractiveTrainer(
                    model_dir=self.model_dir,
                    data_dir=self.data_dir
                )
                
                # Set model reference for preview generation
                if hasattr(self.trainer, 'model'):
                    self.model = self.trainer.model
                
                # Update config
                self.trainer.config['batch_size'] = self.batch_size_var.get()
                self.trainer.config['epochs'] = self.epochs_var.get()
                self.trainer.config['feedback_interval'] = self.feedback_var.get()
                self.trainer.config['save_interval'] = self.save_var.get()
                self.trainer.config['learning_rate_g'] = self.lr_g_var.get()
                self.trainer.config['learning_rate_d'] = self.lr_d_var.get()
            
            # Start training thread
            self.total_epochs = self.epochs_var.get()
            self.training_thread = TrainingThread(
                self.trainer,
                self.total_epochs,
                self.message_queue,
                self.control_queue
            )
            self.training_thread.start()
            
            # Update UI
            self.is_training = True
            self.start_btn.config(state='disabled')
            self.pause_btn.config(state='normal')
            self.stop_btn.config(state='normal')
            
            self.log("Training started!")
            
        except Exception as e:
            print(f"ERROR: Failed to start training: {str(e)}")
            import traceback
            traceback.print_exc()
            self.log(f"ERROR: {str(e)}")
    
    def pause_training(self):
        """Pause training"""
        if self.training_thread:
            if self.pause_btn['text'] == 'Pause':
                self.training_thread.pause()
                self.pause_btn.config(text='Resume')
            else:
                self.training_thread.resume()
                self.pause_btn.config(text='Pause')
    
    def stop_training(self):
        """Stop training"""
        if self.training_thread:
            self.training_thread.stop()
            self.is_training = False
            self.start_btn.config(state='normal')
            self.pause_btn.config(state='disabled')
            self.stop_btn.config(state='disabled')
            self.log("Training stopped")
    
    def test_preview_generation(self):
        """Test the preview generation and display - works even without model"""
        self.log("Testing visual preview generation...")
        
        # Set current epoch if not set
        if not hasattr(self, 'current_epoch'):
            self.current_epoch = 0
        
        # Generate samples (will use rule-based if no model)
        samples = self.generate_and_preview_samples(3)
        
        if samples:
            self.update_training_previews(samples)
            self.log(f"✓ Generated {len(samples)} preview samples successfully!")
            self.log("Preview panels should now show the generated bases")
        else:
            self.log("Failed to generate preview samples")
    
    def load_checkpoint(self):
        """Load a model checkpoint"""
        filepath = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if self.trainer is None:
                    self.trainer = InteractiveTrainer(self.model_dir, self.data_dir)
                    if hasattr(self.trainer, 'model'):
                        self.model = self.trainer.model
                
                # Convert to Path and ensure it exists
                checkpoint_path = Path(filepath)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"File not found: {filepath}")
                
                self.log(f"Loading checkpoint: {filepath}")
                self.trainer.gan.load(checkpoint_path)
                # Update model reference after loading
                if hasattr(self.trainer, 'model'):
                    self.model = self.trainer.model
                elif hasattr(self.trainer, 'gan'):
                    self.model = self.trainer.gan
                self.log(f"✓ Loaded checkpoint: {filepath}")
                print(f"SUCCESS: Checkpoint loaded successfully from {filepath}")
            except Exception as e:
                print(f"ERROR: Failed to load checkpoint: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def save_model(self):
        """Save current model"""
        if self.trainer is None:
            print("WARNING: No model to save")
            self.log("WARNING: No model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.trainer.gan.save(Path(filepath))
                self.log(f"Saved model: {filepath}")
                print(f"SUCCESS: Model saved successfully to {filepath}")
            except Exception as e:
                print(f"ERROR: Failed to save model: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def generate_test_base(self, num_samples=1):
        """Generate test base with current model"""
        if self.trainer is None:
            print("WARNING: No model loaded. Creating new trainer with untrained model.")
            self.log("WARNING: No model loaded. Creating new trainer with untrained model.")
            self.log("Note: Untrained models produce random noise. Train the model first for meaningful results!")
            # Create trainer so user can test even without training
            self.trainer = InteractiveTrainer(self.model_dir, self.data_dir)
            if hasattr(self.trainer, 'model'):
                self.model = self.trainer.model
        
        try:
            # Create requirements
            requirements = BaseRequirements(
                num_colonists=self.test_colonists.get(),
                num_bedrooms=self.test_bedrooms.get(),
                num_workshops=self.test_workshops.get(),
                has_kitchen=self.test_kitchen.get(),
                has_hospital=self.test_hospital.get(),
                has_recreation=self.test_recreation.get(),
                defense_level=self.test_defense.get(),
                beauty_preference=self.test_beauty.get(),
                efficiency_preference=0.8,
                size_constraint=(100, 100)
            )
            
            # Generate
            self.log(f"Generating {num_samples} base(s)...")
            bases = self.trainer.gan.generate(requirements, num_samples)
            
            # Store and display
            self.current_generated_base = bases[0]
            self.current_requirements = requirements
            
            # Display using shared method
            self.display_generated_base(bases[0])
            
            # Check if base is empty (untrained model)
            unique_values = np.unique(bases[0])
            if len(unique_values) <= 2:  # Only 1-2 different cell types
                self.log("⚠️ Generated base appears mostly empty - model needs training!")
                self.log("Tip: Use 'Generate Rule-Based' button for immediate results without training")
                self.log("Or click 'Start Training' in the Training tab to train the AI model first")
            else:
                self.log(f"Generated {num_samples} base(s) successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to generate base: {str(e)}")
            import traceback
            traceback.print_exc()
            self.log(f"ERROR: {str(e)}")
    
    def submit_rating(self):
        """Submit rating for generated base"""
        if not hasattr(self, 'current_generated_base'):
            print("WARNING: No base to rate. Generate a base first.")
            self.log("WARNING: No base to rate. Generate a base first.")
            return
        
        rating = self.rating_var.get() / 10.0
        
        # Save feedback
        if self.trainer:
            self.trainer.feedback_collector.add_feedback(
                self.current_generated_base,
                self.current_requirements,
                rating,
                f"GUI rating: {self.rating_var.get()}/10"
            )
            
            self.log(f"Rating submitted: {self.rating_var.get()}/10")
            print(f"SUCCESS: Rating {self.rating_var.get()}/10 saved")
    
    def display_generated_base(self, layout: np.ndarray):
        """Display a generated base in the GUI"""
        try:
            # Create visualization
            visualizer = RealisticBaseVisualizer(scale=8)
            output_path = self.model_dir / f"test_base_{datetime.now():%Y%m%d_%H%M%S}.png"
            visualizer.visualize(layout, str(output_path), title="Generated Base")
            
            # Display in GUI
            img = Image.open(output_path)
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            if hasattr(self, 'image_label'):
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"ERROR displaying base: {str(e)}")
            self.log(f"ERROR displaying base: {str(e)}")
    
    def generate_rule_based(self):
        """Generate base using rule-based system (no training needed)"""
        try:
            from src.generators.realistic_base_generator import RealisticBaseGenerator
            
            self.log("Generating base with rule-based system...")
            
            # Create generator
            generator = RealisticBaseGenerator(128, 128)
            
            # Generate with current requirements
            layout = generator.generate_base(
                num_bedrooms=self.test_bedrooms.get(),
                include_kitchen=self.test_kitchen.get(),
                include_workshop=self.test_workshops.get(),
                include_hospital=self.test_hospital.get()
            )
            
            # Store for rating/saving
            self.current_generated_base = layout
            self.current_requirements = BaseRequirements(
                num_colonists=self.test_colonists.get(),
                num_bedrooms=self.test_bedrooms.get(),
                num_workshops=self.test_workshops.get(),
                has_kitchen=self.test_kitchen.get(),
                has_hospital=self.test_hospital.get(),
                has_recreation=self.test_recreation.get(),
                defense_level=self.test_defense.get(),
                beauty_preference=self.test_beauty.get(),
                efficiency_preference=0.8,
                size_constraint=(128, 128)
            )
            
            # Display
            self.display_generated_base(layout)
            self.log("✓ Rule-based generation complete (no training required)")
            
        except Exception as e:
            print(f"ERROR: Failed to generate rule-based: {str(e)}")
            import traceback
            traceback.print_exc()
            self.log(f"ERROR: {str(e)}")
    
    def save_generated(self):
        """Save generated base"""
        if not hasattr(self, 'current_generated_base'):
            print("WARNING: No base to save")
            self.log("WARNING: No base to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save base",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filepath:
            visualizer = RealisticBaseVisualizer(scale=10)
            visualizer.visualize(self.current_generated_base, filepath, title="Generated Base")
            self.log(f"Saved base: {filepath}")
    
    def test_side_by_side_feedback(self):
        """Test the side-by-side feedback dialog with multiple generated samples"""
        try:
            self.log("Generating multiple samples for side-by-side comparison...")
            
            # Generate multiple samples
            samples_data = []
            num_samples = 6
            
            for i in range(num_samples):
                self.log(f"Generating sample {i+1}/{num_samples}...")
                
                # Generate with different parameters
                if self.model:
                    # Use AI model
                    requirements = BaseRequirements(
                        num_colonists=self.test_colonists.get() + i,
                        num_bedrooms=self.test_bedrooms.get() + (i % 3),
                        num_workshops=max(1, self.test_workshops.get() - (i % 2)),
                        has_kitchen=True,
                        has_hospital=i % 2 == 0,
                        has_recreation=i % 3 == 0,
                        defense_level=i % 3,
                        beauty_preference=0.5 + (i * 0.1),
                        efficiency_preference=0.8,
                        size_constraint=(128, 128)
                    )
                    layout = self.model.generate(requirements)
                else:
                    # Use rule-based generator
                    from src.generators.realistic_base_generator import RealisticBaseGenerator
                    generator = RealisticBaseGenerator(128, 128)
                    layout = generator.generate_base(
                        num_bedrooms=self.test_bedrooms.get() + (i % 3),
                        include_kitchen=True,
                        include_workshop=i % 2 == 0,
                        include_hospital=i % 3 == 0
                    )
                
                # Save image
                image_path = self.model_dir / f"feedback_sample_{i+1}.png"
                visualizer = RealisticBaseVisualizer(scale=8)
                visualizer.visualize(layout, str(image_path), 
                                   title=f"Sample {i+1}: {self.test_bedrooms.get() + (i % 3)} bedrooms")
                
                samples_data.append({
                    'image_path': str(image_path),
                    'description': f"Base with {self.test_bedrooms.get() + (i % 3)} bedrooms, "
                                 f"{'with' if i % 2 == 0 else 'without'} hospital, "
                                 f"defense level {i % 3}"
                })
            
            self.log(f"Generated {num_samples} samples. Opening side-by-side feedback dialog...")
            
            # Show feedback dialog
            def feedback_callback(results):
                self.log(f"Received feedback for {len(results)} samples")
                for r in results:
                    self.log(f"  Sample {r['sample_idx']+1}: Rating {r['rating']*10:.1f}/10")
                avg_rating = sum(r['rating'] for r in results) / len(results)
                self.log(f"Average rating: {avg_rating*10:.1f}/10")
                
                # If model exists, apply feedback
                if self.model and hasattr(self.model, 'apply_feedback'):
                    self.model.apply_feedback(
                        [s.get('layout') for s in samples_data if 'layout' in s],
                        [r['rating'] for r in results]
                    )
                    self.log("Applied feedback to model")
            
            show_side_by_side_feedback(self.root, samples_data, feedback_callback)
            
        except Exception as e:
            self.log(f"ERROR in side-by-side feedback test: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def collect_prefabs(self):
        """Collect data from AlphaPrefabs"""
        try:
            collector = DatasetCollector(self.data_dir)
            examples = collector.prefabs_parser.collect_all()
            
            collector.save_dataset(examples, "prefabs_dataset.pkl")
            
            self.dataset_info.set(f"Collected {len(examples)} examples from AlphaPrefabs")
            self.log(f"Collected {len(examples)} prefab examples")
            
            # Show preview
            preview = f"Dataset: {len(examples)} examples\n\n"
            for i, ex in enumerate(examples[:5]):
                preview += f"Example {i+1}:\n"
                preview += f"  Source: {ex.source}\n"
                preview += f"  Size: {ex.layout.shape}\n"
                preview += f"  Quality: {ex.quality_scores}\n\n"
            
            self.dataset_text.delete(1.0, tk.END)
            self.dataset_text.insert(1.0, preview)
            
        except Exception as e:
            print(f"ERROR: Failed to collect prefabs: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_synthetic(self):
        """Generate synthetic training data"""
        try:
            num = simpledialog.askinteger("Synthetic Data", "Number of examples to generate:", initialvalue=100)
            if num:
                if self.trainer is None:
                    self.trainer = InteractiveTrainer(self.model_dir, self.data_dir)
                    if hasattr(self.trainer, 'model'):
                        self.model = self.trainer.model
                
                self.log(f"Generating {num} synthetic examples...")
                examples = self.trainer.generate_synthetic_data(num)
                
                self.dataset_info.set(f"Generated {len(examples)} synthetic examples")
                self.log(f"Generated {len(examples)} synthetic examples")
                
        except Exception as e:
            print(f"ERROR: Failed to generate synthetic data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_dataset(self):
        """Load dataset from file"""
        filepath = filedialog.askopenfilename(
            title="Load dataset",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                collector = DatasetCollector(self.data_dir)
                examples = collector.load_dataset(Path(filepath).name)
                
                self.dataset_info.set(f"Loaded {len(examples)} examples")
                self.log(f"Loaded dataset: {len(examples)} examples")
                
            except Exception as e:
                print(f"ERROR: Failed to load dataset: {str(e)}")
                import traceback
                traceback.print_exc()
    
    
    def launch_reddit_scraper(self):
        """Launch the Reddit scraper tool"""
        try:
            import subprocess
            from tkinter import messagebox
            
            self.log("Launching Reddit scraper...")
            
            # Ask user for parameters
            result = messagebox.askyesno(
                "Reddit Scraper", 
                "This will launch the Reddit scraper to collect base designs.\n\n"
                "It will search:\n"
                "• r/RimWorld - Main community subreddit\n"
                "• r/RimWorldPorn - Dedicated to colony screenshots\n\n"
                "This may take several minutes.\n\n"
                "Continue?"
            )
            
            if result:
                # Run the scraper in background
                self.log("Starting Reddit scraper (check console for progress)...")
                subprocess.Popen([
                    "poetry", "run", "python", "reddit_scraper.py",
                    "--max-posts", "50",
                    "--time", "month"
                ])
                
                self.log("Reddit scraper launched. Results will be in data/reddit_bases/")
                self.log("Refresh datasets when scraping is complete.")
                
                # Add refresh button hint
                messagebox.showinfo(
                    "Scraper Running",
                    "The Reddit scraper is now running in the background.\n\n"
                    "Check the console window for progress.\n"
                    "When complete, restart this GUI to load the new data."
                )
        except Exception as e:
            self.log(f"Error launching Reddit scraper: {e}")
            print(f"ERROR: Failed to launch Reddit scraper: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_dataset(self):
        """Export dataset"""
        print("INFO: Dataset export not yet implemented")
        self.log("INFO: Dataset export not yet implemented")
    
    def update_plots(self):
        """Update monitoring plots"""
        if not self.metrics_history['epochs']:
            return
        
        # Clear plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Generator loss
        self.axes[0, 0].plot(self.metrics_history['epochs'], self.metrics_history['g_loss'], 'b-')
        self.axes[0, 0].set_title('Generator Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True)
        
        # Discriminator loss
        self.axes[0, 1].plot(self.metrics_history['epochs'], self.metrics_history['d_loss'], 'r-')
        self.axes[0, 1].set_title('Discriminator Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True)
        
        # Quality score
        self.axes[1, 0].plot(self.metrics_history['epochs'], self.metrics_history['quality'], 'g-')
        self.axes[1, 0].set_title('Quality Score')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Score')
        self.axes[1, 0].grid(True)
        
        # GPU usage
        stats = self.gpu_monitor.get_stats()
        if stats['available']:
            self.axes[1, 1].bar(['Used', 'Free'], 
                               [stats['memory_used'], stats['memory_total'] - stats['memory_used']],
                               color=['red', 'green'])
            self.axes[1, 1].set_title(f'GPU Memory ({stats["name"]})')
            self.axes[1, 1].set_ylabel('Memory (GB)')
        
        self.canvas.draw()
    
    def save_plots(self):
        """Save monitoring plots"""
        filepath = filedialog.asksaveasfilename(
            title="Save plots",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filepath:
            self.fig.savefig(filepath, dpi=150)
            self.log(f"Saved plots: {filepath}")
    
    def show_feedback_dialog(self, samples_data):
        """Show dialog for rating generated samples during training"""
        # Use the new side-by-side feedback dialog
        def feedback_callback(results):
            # Send feedback back to training thread
            if self.training_thread and hasattr(self.training_thread, 'feedback_response'):
                self.training_thread.feedback_response = results
                self.training_thread.feedback_event.set()
            self.log(f"Submitted feedback for {len(results)} samples with average rating: {sum(r['rating'] for r in results)/len(results):.2f}")
        
        # Show the side-by-side dialog
        show_side_by_side_feedback(self.root, samples_data, feedback_callback)
        return
        
        # Old implementation (kept for reference but not executed)
        feedback_window = tk.Toplevel(self.root)
        feedback_window.title("Rate Generated Samples - Training Feedback")
        feedback_window.geometry("800x600")
        
        # Make it modal
        feedback_window.transient(self.root)
        feedback_window.grab_set()
        
        current_sample = [0]  # Use list to allow modification in nested function
        ratings = []
        
        # Title
        title_label = ttk.Label(feedback_window, text="Please rate these generated bases to improve training", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        
        # Image display
        image_frame = ttk.Frame(feedback_window)
        image_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        image_label = ttk.Label(image_frame)
        image_label.pack()
        
        # Info label
        info_label = ttk.Label(feedback_window, text="")
        info_label.pack(pady=5)
        
        # Rating scale
        rating_frame = ttk.Frame(feedback_window)
        rating_frame.pack(pady=10)
        
        ttk.Label(rating_frame, text="Rating (0-10):").pack(side='left', padx=5)
        rating_var = tk.IntVar(value=5)
        rating_scale = ttk.Scale(rating_frame, from_=0, to=10, variable=rating_var, 
                                orient='horizontal', length=300)
        rating_scale.pack(side='left', padx=5)
        rating_value_label = ttk.Label(rating_frame, text="5")
        rating_value_label.pack(side='left', padx=5)
        
        def update_rating_label(value):
            rating_value_label.config(text=str(int(float(value))))
        
        rating_scale.config(command=update_rating_label)
        
        # Comments
        ttk.Label(feedback_window, text="Comments (optional):").pack()
        comment_text = tk.Text(feedback_window, height=3, width=60)
        comment_text.pack(pady=5)
        
        def show_sample(idx):
            """Display current sample"""
            if idx < len(samples_data):
                sample = samples_data[idx]
                # Load and display image
                try:
                    img = Image.open(sample['image_path'])
                    img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    image_label.config(image=photo)
                    image_label.image = photo
                    
                    info_label.config(text=f"Sample {idx+1}/{len(samples_data)} - {sample['description']}")
                except Exception as e:
                    info_label.config(text=f"Error loading image: {e}")
        
        def submit_rating():
            """Submit rating for current sample and show next"""
            rating = rating_var.get() / 10.0
            comments = comment_text.get(1.0, tk.END).strip()
            
            ratings.append({
                'sample_idx': current_sample[0],
                'rating': rating,
                'comments': comments
            })
            
            # Clear comments
            comment_text.delete(1.0, tk.END)
            rating_var.set(5)
            
            # Next sample or finish
            current_sample[0] += 1
            if current_sample[0] < len(samples_data):
                show_sample(current_sample[0])
            else:
                # All samples rated, send feedback back to training thread
                if self.training_thread and hasattr(self.training_thread, 'feedback_response'):
                    self.training_thread.feedback_response = ratings
                    self.training_thread.feedback_event.set()
                self.log(f"Submitted feedback for {len(ratings)} samples")
                feedback_window.destroy()
        
        # Buttons
        button_frame = ttk.Frame(feedback_window)
        button_frame.pack(pady=10)
        
        submit_btn = ttk.Button(button_frame, text="Submit Rating & Next", command=submit_rating)
        submit_btn.pack(side='left', padx=5)
        
        skip_btn = ttk.Button(button_frame, text="Skip All Feedback", 
                             command=lambda: [
                                 setattr(self.training_thread, 'feedback_response', 
                                        [{'rating': 0.5, 'comments': 'Skipped'} for _ in samples_data])
                                        if self.training_thread else None,
                                 self.training_thread.feedback_event.set() if self.training_thread else None,
                                 feedback_window.destroy(),
                                 self.log("Skipped feedback")
                             ])
        skip_btn.pack(side='left', padx=5)
        
        # Show first sample
        show_sample(0)
        
        # Focus window
        feedback_window.focus_force()
    
    def update_training_previews(self, epoch_samples):
        """Update the visual preview panels during training"""
        try:
            if not epoch_samples:
                return
                
            # Sort samples by quality/score
            sorted_samples = sorted(epoch_samples, key=lambda x: x.get('score', 0), reverse=True)
            
            # Update best sample
            if sorted_samples:
                best = sorted_samples[0]
                self.update_preview_image(self.best_image_label, best.get('image_path'))
                self.best_score_label.config(text=f"Score: {best.get('score', 0):.2f}")
            
            # Update worst sample
            if len(sorted_samples) > 1:
                worst = sorted_samples[-1]
                self.update_preview_image(self.worst_image_label, worst.get('image_path'))
                self.worst_score_label.config(text=f"Score: {worst.get('score', 0):.2f}")
            
            # Update latest sample
            if sorted_samples:
                latest = sorted_samples[len(sorted_samples)//2]  # Middle sample
                self.update_preview_image(self.latest_image_label, latest.get('image_path'))
                self.latest_info_label.config(text=f"Epoch: {self.current_epoch}")
                
        except Exception as e:
            self.log(f"Error updating previews: {e}")
    
    def update_preview_image(self, label, image_path):
        """Update a preview image label"""
        try:
            if image_path and Path(image_path).exists():
                img = Image.open(image_path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label.config(image=photo, text="")
                label.image = photo  # Keep reference
            else:
                label.config(text="No image", image="")
        except Exception as e:
            label.config(text=f"Error: {e}", image="")
    
    def generate_and_preview_samples(self, num_samples=3):
        """Generate samples during training and show them in preview"""
        samples = []
        
        try:
            # Use model if available, otherwise use rule-based generator
            if self.model:
                # Model-based generation
                for i in range(num_samples):
                    requirements = BaseRequirements(
                        num_colonists=8 + (i * 2),
                        num_bedrooms=6 + i,
                        num_workshops=2 + (i % 2),
                        has_kitchen=True,
                        has_hospital=i % 2 == 0,
                        has_recreation=True,
                        defense_level=i % 3,
                        beauty_preference=0.5 + (i * 0.1),
                        efficiency_preference=0.8,
                        size_constraint=(128, 128)
                    )
                    
                    layout = self.model.generate(requirements)
                    samples.append(self._create_preview_sample(layout, i))
            else:
                # Rule-based generation for testing/early training
                from src.generators.realistic_base_generator import RealisticBaseGenerator
                
                for i in range(num_samples):
                    generator = RealisticBaseGenerator(128, 128)
                    layout = generator.generate_base(
                        num_bedrooms=5 + i,
                        include_kitchen=True,
                        include_workshop=1 + (i % 2),
                        include_hospital=i == 0,
                        include_storage=True,
                        include_dining=i > 0
                    )
                    samples.append(self._create_preview_sample(layout, i))
                    
        except Exception as e:
            self.log(f"Error generating preview samples: {e}")
            import traceback
            traceback.print_exc()
            
        return samples
    
    def _create_preview_sample(self, layout, index):
        """Create a preview sample from a layout"""
        # Save preview image
        preview_path = self.model_dir / f"epoch_{self.current_epoch}_sample_{index}.png"
        visualizer = RealisticBaseVisualizer(scale=6)
        visualizer.visualize(layout, str(preview_path), 
                           title=f"Epoch {self.current_epoch} Sample {index+1}")
        
        # Calculate simple quality score
        unique_cells = len(np.unique(layout))
        non_empty = np.sum(layout > 0)
        score = (unique_cells / 10.0) + (non_empty / (128 * 128) * 5)
        
        return {
            'image_path': str(preview_path),
            'layout': layout,
            'score': min(10, score),
            'epoch': self.current_epoch
        }
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history = {
            'epochs': [],
            'g_loss': [],
            'd_loss': [],
            'quality': []
        }
        self.update_plots()
        self.log("Cleared metrics history")
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_loop(self):
        """Main update loop"""
        # Update GPU status
        self.update_gpu_status()
        
        # Process messages from training thread
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == 'status':
                    self.log(msg_data)
                elif msg_type == 'epoch':
                    self.current_epoch = msg_data
                    progress = (self.current_epoch / self.total_epochs) * 100
                    self.progress_bar['value'] = progress
                    self.progress_text.set(f"Epoch {self.current_epoch}/{self.total_epochs} ({progress:.1f}%)")
                elif msg_type == 'metrics':
                    self.metrics_history['epochs'].append(self.current_epoch)
                    self.metrics_history['g_loss'].append(msg_data['g_loss'])
                    self.metrics_history['d_loss'].append(msg_data['d_loss'])
                    self.metrics_history['quality'].append(msg_data['quality'])
                    self.update_plots()
                elif msg_type == 'generate_previews':
                    # Generate and show preview samples
                    epoch = msg_data
                    self.log(f"Updating visual previews for epoch {epoch}...")
                    samples = self.generate_and_preview_samples(3)
                    if samples:
                        self.update_training_previews(samples)
                elif msg_type == 'complete':
                    self.log(msg_data)
                    self.is_training = False
                    self.start_btn.config(state='normal')
                    self.pause_btn.config(state='disabled')
                    self.stop_btn.config(state='disabled')
                    print(f"TRAINING COMPLETE: {msg_data}")
                    self.log(f"Training complete: {msg_data}")
                elif msg_type == 'feedback_request':
                    # Show feedback dialog for generated samples
                    self.log("Feedback requested - showing dialog...")
                    self.show_feedback_dialog(msg_data)
                elif msg_type == 'error':
                    self.log(f"ERROR: {msg_data}")
                    print(f"TRAINING ERROR: {msg_data}")
                    self.log(f"Training error: {msg_data}")
                    self.stop_training()
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(1000, self.update_loop)
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    """Main entry point"""
    print("\nStarting ML Training GUI...")
    print("Debug output enabled - watch console for messages\n")
    
    try:
        app = MLTrainingGUI()
        print("GUI initialized successfully")
        app.run()
    except Exception as e:
        print(f"\n{'='*60}")
        print("FATAL ERROR: GUI crashed during initialization")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        
        # Try to show error in messagebox if possible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Fatal Error", f"GUI failed to start:\n\n{str(e)}\n\nCheck console for details.")
        except Exception:
            pass
        
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
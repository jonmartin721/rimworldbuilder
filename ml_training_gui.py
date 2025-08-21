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
import json
from datetime import datetime
import psutil
import GPUtil
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.base_gan_model import BaseGAN, BaseRequirements
from src.ml.train_base_model import InteractiveTrainer
from src.ml.dataset_collector import DatasetCollector
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
        except:
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
        try:
            # Check for existing prefabs dataset
            prefabs_path = self.data_dir / "prefabs_dataset.pkl"
            if prefabs_path.exists():
                import pickle
                with open(prefabs_path, 'rb') as f:
                    examples = pickle.load(f)
                self.log(f"Loaded existing prefabs dataset: {len(examples)} examples")
                
                # Update dataset info
                self.dataset_info.set(f"Prefabs dataset loaded: {len(examples)} examples")
            else:
                # Auto-collect prefabs if they don't exist
                self.log("No prefabs dataset found. Collecting from AlphaPrefabs...")
                self.collect_prefabs()
                
        except Exception as e:
            self.log(f"Error loading initial datasets: {e}")
    
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
        self.feedback_var = tk.IntVar(value=20)
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
        
        # Progress
        progress_frame = ttk.LabelFrame(tab, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_text = tk.StringVar(value="Not training")
        ttk.Label(progress_frame, textvariable=self.progress_text).pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, font=('Consolas', 9))
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
        ttk.Button(control_frame, text="Collect Online Designs", command=self.collect_online_designs).pack(side='left', padx=5)
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
                
                # Convert to Path and ensure it exists
                checkpoint_path = Path(filepath)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"File not found: {filepath}")
                
                self.log(f"Loading checkpoint: {filepath}")
                self.trainer.gan.load(checkpoint_path)
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
    
    def collect_online_designs(self):
        """Collect base designs from online sources"""
        try:
            from src.ml.online_design_collector import OnlineDesignCollector
            
            self.log("Collecting online designs from Reddit...")
            collector = OnlineDesignCollector(self.data_dir)
            designs = collector.collect_all()
            
            if designs:
                self.dataset_info.set(f"Collected {len(designs)} online designs")
                self.log(f"Found {len(designs)} base designs online")
                
                # Show preview
                preview = f"Online Designs: {len(designs)} found\n\n"
                for i, design in enumerate(designs[:5]):
                    preview += f"Design {i+1}: {design.title}\n"
                    preview += f"  Source: {design.source}\n"
                    preview += f"  Score: {design.score} upvotes\n"
                    if design.colonist_count:
                        preview += f"  Colonists: {design.colonist_count}\n"
                    preview += "\n"
                
                self.dataset_text.delete(1.0, tk.END)
                self.dataset_text.insert(1.0, preview)
            else:
                self.log("No online designs found")
                
        except Exception as e:
            print(f"ERROR: Failed to collect online designs: {str(e)}")
            import traceback
            traceback.print_exc()
            self.log(f"ERROR: {str(e)}")
    
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
        # Create top-level window for feedback
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
        print(f"FATAL ERROR: GUI crashed during initialization")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        
        # Try to show error in messagebox if possible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Fatal Error", f"GUI failed to start:\n\n{str(e)}\n\nCheck console for details.")
        except:
            pass
        
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
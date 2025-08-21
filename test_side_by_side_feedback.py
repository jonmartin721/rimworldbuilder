#!/usr/bin/env python3
"""
Test script for the side-by-side feedback system
"""

import tkinter as tk
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.side_by_side_feedback import show_side_by_side_feedback
from src.generators.realistic_base_generator import RealisticBaseGenerator
from src.visualization.realistic_visualizer import RealisticBaseVisualizer
import numpy as np


def generate_test_samples(num_samples=6):
    """Generate test base samples"""
    samples_data = []
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} test samples...")
    
    for i in range(num_samples):
        print(f"  Generating sample {i+1}/{num_samples}...")
        
        # Generate base with varying parameters
        generator = RealisticBaseGenerator(128, 128)
        layout = generator.generate_base(
            num_bedrooms=4 + (i % 4),
            include_kitchen=True,
            include_workshop=1 + (i % 2),  # Number of workshops
            include_hospital=i % 3 == 0,
            include_storage=True,
            include_dining=i > 2
        )
        
        # Save visualization
        image_path = output_dir / f"test_feedback_{i+1}.png"
        visualizer = RealisticBaseVisualizer(scale=8)
        
        # Create title with details
        details = []
        details.append(f"{4 + (i % 4)} bedrooms")
        details.append(f"{1 + (i % 2)} workshops")
        if i % 3 == 0:
            details.append("hospital")
        if i > 2:
            details.append("dining")
        
        title = f"Sample {i+1}: {', '.join(details)}"
        
        visualizer.visualize(layout, str(image_path), title=title)
        
        samples_data.append({
            'image_path': str(image_path),
            'description': f"Base design with {', '.join(details)}"
        })
    
    return samples_data


def main():
    """Main test function"""
    print("Testing Side-by-Side Feedback System")
    print("=" * 50)
    
    # Create root window (hidden)
    root = tk.Tk()
    root.title("Side-by-Side Feedback Test")
    root.geometry("400x200")
    
    # Add a label
    label = tk.Label(root, text="Side-by-Side Feedback Test\n\nGenerating samples...", 
                     font=('Arial', 12))
    label.pack(pady=20)
    
    # Generate button
    def run_test():
        label.config(text="Generating test samples...")
        root.update()
        
        # Generate samples
        samples = generate_test_samples(6)
        
        label.config(text="Opening feedback dialog...")
        root.update()
        
        # Callback for results
        def on_feedback(results):
            print("\n" + "=" * 50)
            print("FEEDBACK RESULTS:")
            print("=" * 50)
            for r in results:
                print(f"Sample {r['sample_idx']+1}:")
                print(f"  Rating: {r['rating']*10:.1f}/10")
                if r.get('comments'):
                    print(f"  Comments: {r['comments']}")
            
            avg = sum(r['rating'] for r in results) / len(results)
            print(f"\nAverage Rating: {avg*10:.1f}/10")
            
            label.config(text=f"Test Complete!\n\nAverage Rating: {avg*10:.1f}/10")
        
        # Show dialog
        show_side_by_side_feedback(root, samples, on_feedback)
    
    button = tk.Button(root, text="Run Feedback Test", command=run_test,
                      bg='#3498DB', fg='white', font=('Arial', 10, 'bold'),
                      padx=20, pady=10)
    button.pack(pady=20)
    
    # Exit button
    tk.Button(root, text="Exit", command=root.quit, padx=20, pady=5).pack()
    
    print("Click 'Run Feedback Test' to begin")
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Side-by-side feedback dialog for ranking multiple generated bases simultaneously.
Allows quick comparison and ranking of multiple samples at once.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class SideBySideFeedbackDialog:
    """Enhanced feedback dialog showing all samples side-by-side for easy ranking"""
    
    def __init__(self, parent, samples_data: List[Dict[str, Any]], callback=None):
        """
        Initialize the side-by-side feedback dialog
        
        Args:
            parent: Parent tkinter window
            samples_data: List of sample dictionaries with 'image_path' and 'description'
            callback: Optional callback function to receive ratings
        """
        self.parent = parent
        self.samples_data = samples_data
        self.callback = callback
        self.ratings = {}
        self.sample_widgets = []
        
        # Create dialog window
        self.window = tk.Toplevel(parent)
        self.window.title("Side-by-Side Base Ranking")
        
        # Calculate window size based on number of samples
        num_samples = len(samples_data)
        columns = min(4, num_samples)  # Max 4 columns
        rows = (num_samples + columns - 1) // columns
        
        # Window dimensions
        sample_width = 300
        sample_height = 400
        window_width = min(1800, columns * (sample_width + 20) + 40)
        window_height = min(900, rows * (sample_height + 100) + 150)
        
        self.window.geometry(f"{window_width}x{window_height}")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.setup_ui(columns, rows)
        self.load_samples()
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.window.winfo_screenheight() // 2) - (window_height // 2)
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def setup_ui(self, columns: int, rows: int):
        """Setup the UI layout"""
        # Title and instructions
        title_frame = ttk.Frame(self.window)
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = ttk.Label(
            title_frame,
            text="Rank Generated Bases Side-by-Side",
            font=('Arial', 14, 'bold')
        )
        title_label.pack()
        
        instructions = ttk.Label(
            title_frame,
            text="Rate each base from 0-10. Use quick rank buttons for relative ranking.",
            font=('Arial', 10)
        )
        instructions.pack(pady=5)
        
        # Quick ranking buttons
        quick_rank_frame = ttk.Frame(title_frame)
        quick_rank_frame.pack(pady=10)
        
        ttk.Button(
            quick_rank_frame,
            text="Auto-Rank by Visual Quality",
            command=self.auto_rank_visual
        ).pack(side='left', padx=5)
        
        ttk.Button(
            quick_rank_frame,
            text="Set All to Average (5)",
            command=lambda: self.set_all_ratings(5)
        ).pack(side='left', padx=5)
        
        ttk.Button(
            quick_rank_frame,
            text="Clear All Ratings",
            command=self.clear_ratings
        ).pack(side='left', padx=5)
        
        # Main content with scrollbar
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Create canvas with scrollbar for samples
        canvas = tk.Canvas(main_frame, bg='white')
        scrollbar_y = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(self.window, orient='horizontal', command=canvas.xview)
        
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side='right', fill='y')
        scrollbar_x.pack(side='bottom', fill='x', padx=20)
        canvas.pack(side='left', fill='both', expand=True)
        
        # Frame inside canvas for samples
        self.samples_frame = ttk.Frame(canvas)
        canvas.create_window(0, 0, anchor='nw', window=self.samples_frame)
        
        # Configure canvas scrolling
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox('all'))
        
        self.samples_frame.bind('<Configure>', configure_scroll)
        
        # Create sample widgets in grid
        for idx, sample in enumerate(self.samples_data):
            row = idx // columns
            col = idx % columns
            self.create_sample_widget(self.samples_frame, idx, sample, row, col)
        
        # Bottom action buttons
        action_frame = ttk.Frame(self.window)
        action_frame.pack(fill='x', pady=20)
        
        # Statistics label
        self.stats_label = ttk.Label(action_frame, text="", font=('Arial', 10))
        self.stats_label.pack(side='left', padx=20)
        
        # Action buttons
        button_frame = ttk.Frame(action_frame)
        button_frame.pack(side='right', padx=20)
        
        ttk.Button(
            button_frame,
            text="Submit Rankings",
            command=self.submit_rankings,
            style='Accent.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel
        ).pack(side='left', padx=5)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        self.update_statistics()
    
    def create_sample_widget(self, parent, idx: int, sample: Dict, row: int, col: int):
        """Create widget for a single sample"""
        frame = ttk.LabelFrame(parent, text=f"Sample {idx + 1}", padding=10)
        frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        # Store widget reference
        widget_data = {
            'frame': frame,
            'idx': idx,
            'sample': sample
        }
        
        # Image display
        image_label = ttk.Label(frame)
        image_label.pack(pady=5)
        widget_data['image_label'] = image_label
        
        # Description
        desc_text = sample.get('description', f'Base {idx + 1}')
        if len(desc_text) > 50:
            desc_text = desc_text[:47] + "..."
        desc_label = ttk.Label(frame, text=desc_text, wraplength=250)
        desc_label.pack(pady=5)
        
        # Rating frame
        rating_frame = ttk.Frame(frame)
        rating_frame.pack(pady=5)
        
        ttk.Label(rating_frame, text="Rating:").pack(side='left', padx=5)
        
        # Rating scale
        rating_var = tk.DoubleVar(value=5.0)
        rating_scale = ttk.Scale(
            rating_frame,
            from_=0,
            to=10,
            variable=rating_var,
            orient='horizontal',
            length=150
        )
        rating_scale.pack(side='left', padx=5)
        
        # Rating value label
        rating_label = ttk.Label(rating_frame, text="5.0")
        rating_label.pack(side='left', padx=5)
        
        def update_rating(value):
            val = round(float(value), 1)
            rating_label.config(text=str(val))
            self.ratings[idx] = val
            self.update_statistics()
            # Update frame color based on rating
            if val >= 8:
                frame.configure(style='Good.TLabelframe')
            elif val >= 6:
                frame.configure(style='Medium.TLabelframe')
            elif val >= 4:
                frame.configure(style='TLabelframe')
            else:
                frame.configure(style='Poor.TLabelframe')
        
        rating_scale.config(command=update_rating)
        widget_data['rating_var'] = rating_var
        widget_data['rating_scale'] = rating_scale
        
        # Quick rating buttons
        quick_frame = ttk.Frame(frame)
        quick_frame.pack(pady=5)
        
        for val, text, color in [(0, "Bad", "red"), (5, "OK", "yellow"), (10, "Great", "green")]:
            btn = tk.Button(
                quick_frame,
                text=text,
                width=6,
                bg=color,
                fg='white' if color != 'yellow' else 'black',
                command=lambda v=val, rv=rating_var: [rv.set(v), update_rating(v)]
            )
            btn.pack(side='left', padx=2)
        
        # Comments field (collapsed by default)
        comment_frame = ttk.Frame(frame)
        comment_frame.pack(fill='x', pady=5)
        
        comment_text = tk.Text(comment_frame, height=2, width=30)
        comment_text.pack()
        comment_text.insert('1.0', sample.get('comments', ''))
        widget_data['comment_text'] = comment_text
        
        # Store widget data
        self.sample_widgets.append(widget_data)
        
        # Initialize rating
        self.ratings[idx] = 5.0
    
    def load_samples(self):
        """Load and display sample images"""
        for widget_data in self.sample_widgets:
            try:
                # Load image
                image_path = widget_data['sample'].get('image_path')
                if image_path and Path(image_path).exists():
                    img = Image.open(image_path)
                    # Resize to fit
                    img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    widget_data['image_label'].config(image=photo)
                    widget_data['image_label'].image = photo
                else:
                    # Create placeholder
                    img = Image.new('RGB', (250, 250), color='lightgray')
                    photo = ImageTk.PhotoImage(img)
                    widget_data['image_label'].config(image=photo)
                    widget_data['image_label'].image = photo
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def auto_rank_visual(self):
        """Auto-rank based on visual complexity and structure"""
        # Simple heuristic ranking based on image analysis
        rankings = []
        
        for widget_data in self.sample_widgets:
            idx = widget_data['idx']
            try:
                image_path = widget_data['sample'].get('image_path')
                if image_path and Path(image_path).exists():
                    img = Image.open(image_path)
                    # Simple metrics for ranking
                    pixels = img.convert('L').tobytes()
                    # Variance as a proxy for detail/complexity
                    import numpy as np
                    arr = np.frombuffer(pixels, dtype=np.uint8)
                    variance = np.var(arr)
                    # Non-empty pixels
                    non_empty = np.sum(arr > 10)
                    score = (variance / 1000) + (non_empty / len(arr)) * 5
                    rankings.append((idx, min(10, max(0, score))))
                else:
                    rankings.append((idx, 5.0))
            except Exception:
                rankings.append((idx, 5.0))
        
        # Sort and assign relative rankings
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute rankings from 2 to 9
        if len(rankings) > 1:
            for i, (idx, _) in enumerate(rankings):
                # Linear distribution from 9 to 2
                rating = 9 - (7 * i / (len(rankings) - 1))
                self.sample_widgets[idx]['rating_var'].set(rating)
                self.ratings[idx] = rating
        
        self.update_statistics()
        messagebox.showinfo("Auto-Rank", "Bases ranked by visual complexity and structure")
    
    def set_all_ratings(self, value: float):
        """Set all ratings to the same value"""
        for widget_data in self.sample_widgets:
            widget_data['rating_var'].set(value)
            self.ratings[widget_data['idx']] = value
        self.update_statistics()
    
    def clear_ratings(self):
        """Clear all ratings to default"""
        self.set_all_ratings(5.0)
    
    def update_statistics(self):
        """Update statistics display"""
        if self.ratings:
            values = list(self.ratings.values())
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            stats_text = (f"Average: {avg:.1f} | "
                         f"Min: {min_val:.1f} | "
                         f"Max: {max_val:.1f} | "
                         f"Rated: {len([v for v in values if v != 5.0])}/{len(values)}")
            self.stats_label.config(text=stats_text)
    
    def submit_rankings(self):
        """Submit all rankings"""
        results = []
        for widget_data in self.sample_widgets:
            idx = widget_data['idx']
            rating = self.ratings.get(idx, 5.0) / 10.0  # Normalize to 0-1
            comments = widget_data['comment_text'].get('1.0', tk.END).strip()
            
            results.append({
                'sample_idx': idx,
                'rating': rating,
                'normalized_rating': rating,
                'comments': comments,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save to file for analysis
        feedback_file = Path("feedback_history.json")
        history = []
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass
        
        history.append({
            'session': datetime.now().isoformat(),
            'num_samples': len(results),
            'ratings': results,
            'average_rating': sum(r['rating'] for r in results) / len(results) if results else 0
        })
        
        with open(feedback_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Call callback if provided
        if self.callback:
            self.callback(results)
        
        messagebox.showinfo("Success", f"Submitted rankings for {len(results)} samples")
        self.window.destroy()
    
    def cancel(self):
        """Cancel without saving"""
        if messagebox.askyesno("Cancel", "Cancel without saving rankings?"):
            self.window.destroy()


def show_side_by_side_feedback(parent, samples_data: List[Dict[str, Any]], callback=None):
    """
    Convenience function to show the side-by-side feedback dialog
    
    Args:
        parent: Parent tkinter window
        samples_data: List of sample dictionaries
        callback: Optional callback to receive results
    
    Returns:
        The dialog instance
    """
    dialog = SideBySideFeedbackDialog(parent, samples_data, callback)
    return dialog


if __name__ == "__main__":
    # Test the dialog
    root = tk.Tk()
    root.withdraw()
    
    # Create dummy samples
    test_samples = []
    for i in range(6):
        test_samples.append({
            'image_path': f"output/example{i+1}_defensive.png",
            'description': f"Test base design {i+1} with various features"
        })
    
    def on_feedback(results):
        print(f"Received {len(results)} ratings:")
        for r in results:
            print(f"  Sample {r['sample_idx']}: {r['rating']:.2f} - {r['comments']}")
    
    dialog = show_side_by_side_feedback(root, test_samples, on_feedback)
    root.mainloop()
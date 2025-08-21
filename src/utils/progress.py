"""
Progress indicators and logging utilities for better user feedback.
"""

import sys
import time
import threading
from typing import Optional
from contextlib import contextmanager

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # Fallback to default encoding

# Import symbols with fallbacks
try:
    from src.utils.symbols import (
        SUCCESS,
        FAILURE,
        WARNING,
        CHECK,
        CROSS,
        HOURGLASS,
        TIMER,
        ARROW_RIGHT,
        BULLET,
    )
except ImportError:
    # Fallback if symbols module not available
    SUCCESS = "[OK]"
    FAILURE = "[FAIL]"
    WARNING = "[WARN]"
    CHECK = "[v]"
    CROSS = "[x]"
    HOURGLASS = "[...]"
    TIMER = "[TIME]"
    ARROW_RIGHT = "->"
    BULLET = "*"


class Spinner:
    """Animated spinner for long-running operations"""

    def __init__(self, message: str = "Processing"):
        self.message = message
        # Use ASCII chars on Windows if Unicode not supported
        try:
            "⠋".encode(sys.stdout.encoding)
            self.chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        except (UnicodeEncodeError, AttributeError):
            self.chars = ["|", "/", "-", "\\"]
        self.delay = 0.1
        self.running = False
        self.thread = None

    def _spin(self):
        """Internal spinning animation"""
        idx = 0
        while self.running:
            char = self.chars[idx % len(self.chars)]
            sys.stdout.write(f"\r{char} {self.message}...")
            sys.stdout.flush()
            time.sleep(self.delay)
            idx += 1

    def start(self):
        """Start the spinner"""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, final_message: Optional[str] = None):
        """Stop the spinner"""
        self.running = False
        if self.thread:
            self.thread.join()

        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")

        if final_message:
            print(final_message)
        sys.stdout.flush()


class ProgressBar:
    """Simple progress bar for iterative operations"""

    def __init__(self, total: int, width: int = 40, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, current: Optional[int] = None, message: str = ""):
        """Update progress bar"""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        # Calculate percentage
        percent = min(100, int(100 * self.current / self.total))

        # Calculate bar fill
        filled = int(self.width * self.current / self.total)
        # Use ASCII chars for better compatibility
        try:
            "█".encode(sys.stdout.encoding)
            bar = "█" * filled + "░" * (self.width - filled)
        except (UnicodeEncodeError, AttributeError):
            bar = "#" * filled + "-" * (self.width - filled)

        # Display
        sys.stdout.write(f"\r{self.prefix}: [{bar}] {percent}% {message}")
        sys.stdout.flush()

        if self.current >= self.total:
            print()  # New line when complete

    def finish(self, message: str = "Complete!"):
        """Finish the progress bar"""
        self.current = self.total
        self.update(message=message)


class StepProgress:
    """Progress indicator for multi-step operations"""

    def __init__(self, steps: list):
        self.steps = steps
        self.current_step = 0
        self.total_steps = len(steps)

    def start_step(self, step_name: Optional[str] = None):
        """Start a new step"""
        if step_name is None and self.current_step < self.total_steps:
            step_name = self.steps[self.current_step]

        print(f"\n[{self.current_step + 1}/{self.total_steps}] {step_name}...")
        self.current_step += 1

    def complete_step(self, message: str = None):
        """Mark current step as complete"""
        if message is None:
            message = CHECK
        print(f"    {message}")

    def finish(self):
        """Finish all steps"""
        print(f"\n{SUCCESS} All {self.total_steps} steps completed!")


@contextmanager
def spinner(message: str = "Processing"):
    """Context manager for spinner"""
    s = Spinner(message)
    s.start()
    try:
        yield s
    finally:
        s.stop()


@contextmanager
def step_context(step_name: str):
    """Context manager for a single step with timing"""
    print(f"\n{HOURGLASS} {step_name}...")
    start_time = time.time()

    try:
        yield
        elapsed = time.time() - start_time
        print(f"{CHECK} {step_name} completed in {elapsed:.2f}s")
    except Exception as e:
        print(f"{CROSS} {step_name} failed: {e}")
        raise


def log_section(title: str, width: int = 60):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def log_subsection(title: str, width: int = 40):
    """Print a formatted subsection header"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def log_item(label: str, value: str, indent: int = 2):
    """Log a labeled item"""
    print(" " * indent + f"{BULLET} {label}: {value}")


def log_list(items: list, indent: int = 4):
    """Log a list of items"""
    for item in items:
        print(" " * indent + f"- {item}")


def format_time(seconds: float) -> str:
    """Format time duration nicely"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


class GenerationLogger:
    """Specialized logger for generation operations"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None
        self.step_times = []

    def start(self, title: str = "Generation"):
        """Start logging a generation"""
        self.start_time = time.time()
        if self.verbose:
            log_section(title)

    def step(self, message: str):
        """Log a generation step"""
        if self.verbose:
            step_time = time.time()
            if self.step_times:
                last_time = self.step_times[-1]
                elapsed = format_time(step_time - last_time)
                print(f"  {CHECK} Previous step: {elapsed}")

            print(f"\n{ARROW_RIGHT} {message}")
            self.step_times.append(step_time)

    def detail(self, message: str):
        """Log a detail within a step"""
        if self.verbose:
            print(f"    {message}")

    def success(self, message: str):
        """Log a success message"""
        if self.verbose:
            print(f"  {SUCCESS} {message}")

    def warning(self, message: str):
        """Log a warning message"""
        if self.verbose:
            print(f"  {WARNING} {message}")

    def error(self, message: str):
        """Log an error message"""
        print(f"  {FAILURE} {message}")

    def finish(self):
        """Finish logging"""
        if self.start_time and self.verbose:
            total_time = time.time() - self.start_time
            print(f"\n{TIMER} Total time: {format_time(total_time)}")
            log_section("Generation Complete", 60)

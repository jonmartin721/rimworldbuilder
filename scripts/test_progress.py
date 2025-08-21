#!/usr/bin/env python3
"""
Test script to demonstrate progress indicators
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.progress import (
    Spinner,
    ProgressBar,
    StepProgress,
    spinner,
    step_context,
    GenerationLogger,
    log_section,
    log_subsection,
    log_item,
)


def test_spinner():
    """Test spinner functionality"""
    print("\n=== Testing Spinner ===")

    s = Spinner("Loading data")
    s.start()
    time.sleep(2)
    s.stop("✅ Data loaded!")

    # Context manager version
    with spinner("Processing files"):
        time.sleep(2)
    print("✅ Files processed!")


def test_progress_bar():
    """Test progress bar"""
    print("\n=== Testing Progress Bar ===")

    items = ["file1.txt", "file2.txt", "file3.txt", "file4.txt", "file5.txt"]
    progress = ProgressBar(len(items), prefix="Processing")

    for i, item in enumerate(items):
        progress.update(i, f"Processing {item}")
        time.sleep(0.5)

    progress.finish("All files processed!")


def test_step_progress():
    """Test step progress"""
    print("\n=== Testing Step Progress ===")

    steps = [
        "Loading configuration",
        "Parsing save file",
        "Analyzing buildings",
        "Generating base",
        "Saving output",
    ]

    progress = StepProgress(steps)

    for step in steps:
        progress.start_step()
        time.sleep(0.5)
        progress.complete_step("Done")

    progress.finish()


def test_generation_logger():
    """Test generation logger"""
    print("\n=== Testing Generation Logger ===")

    logger = GenerationLogger()
    logger.start("TEST GENERATION")

    logger.step("Analyzing requirements")
    logger.detail("Found 5 bedrooms needed")
    logger.detail("Kitchen required")
    time.sleep(0.5)

    logger.step("Matching prefabs")
    logger.detail("Searching prefab library...")
    logger.detail("Found 12 suitable prefabs")
    time.sleep(0.5)

    logger.step("Placing rooms")
    logger.success("Placed bedroom 1")
    logger.success("Placed bedroom 2")
    logger.warning("Could not place bedroom 3")
    time.sleep(0.5)

    logger.step("Adding decorations")
    logger.detail("Filling empty spaces")
    time.sleep(0.5)

    logger.finish()


def test_formatting():
    """Test formatting functions"""
    print("\n=== Testing Formatting ===")

    log_section("MAIN SECTION", 50)

    log_subsection("Configuration")
    log_item("Width", "60 tiles")
    log_item("Height", "60 tiles")
    log_item("Style", "Defensive")

    log_subsection("Requirements")
    log_item("Colonists", "8")
    log_item("Defense Level", "High")

    # Test step context
    with step_context("Loading prefabs"):
        time.sleep(0.5)

    with step_context("Generating base"):
        time.sleep(0.5)


def main():
    """Run all tests"""
    print("=" * 60)
    print("PROGRESS INDICATOR TESTS".center(60))
    print("=" * 60)

    test_spinner()
    test_progress_bar()
    test_step_progress()
    test_generation_logger()
    test_formatting()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE".center(60))
    print("=" * 60)


if __name__ == "__main__":
    main()

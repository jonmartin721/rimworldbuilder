#!/bin/bash

echo "============================================"
echo "      RimWorld Base Assistant Launcher"
echo "============================================"
echo ""
echo "Choose an option:"
echo "1. Command-Line Interface (Recommended)"
echo "2. Graphical User Interface"
echo "3. Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting CLI..."
        poetry run python rimworld_assistant.py
        ;;
    2)
        echo ""
        echo "Starting GUI..."
        poetry run python rimworld_assistant_gui.py
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run ./start.sh again."
        ;;
esac
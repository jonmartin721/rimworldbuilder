#!/bin/bash

echo "============================================"
echo "      RimWorld Base Assistant Launcher"
echo "============================================"
echo ""
echo "Choose an option:"
echo "1. Enhanced GUI with Best Practices (Recommended)"
echo "2. Command-Line Interface"
echo "3. Classic GUI"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting Enhanced GUI with RimWorld Best Practices..."
        poetry run python rimworld_assistant_gui_v2.py
        ;;
    2)
        echo ""
        echo "Starting CLI..."
        poetry run python rimworld_assistant.py
        ;;
    3)
        echo ""
        echo "Starting Classic GUI..."
        poetry run python rimworld_assistant_gui.py
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run ./start.sh again."
        ;;
esac
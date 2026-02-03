#!/bin/bash
# Quick Start Guide - Run this to get started quickly

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║     JetRacer Autonomous System - Quick Start Guide       ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Setup
echo "Step 1: Running setup..."
echo "----------------------------------------"
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Are you in the right directory?"
    exit 1
fi

echo "This will install dependencies and create a virtual environment. Continue? (y/n)"
read -r answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    ./setup.sh
else
    echo "Skipping setup. Make sure dependencies are installed!"
    echo "Checking for virtual environment..."
    if [ -d ".venv" ]; then
        echo "✓ Virtual environment exists"
        source .venv/bin/activate
    else
        echo "⚠ No virtual environment found. Some packages may not work."
    fi
fi

echo ""

# Step 2: Test system
echo "Step 2: Testing system components..."
echo "----------------------------------------"
python3 preflight_check.py

test_result=$?

echo ""

# Step 3: Start Docker
echo "Step 3: Starting Docker container..."
echo "----------------------------------------"
if docker ps --format '{{.Names}}' | grep -q "^jetracer_ros$"; then
    echo "✓ Docker container 'jetracer_ros' is already running"
else
    echo "Starting Docker container..."
    if [ -f ~/start_jetracer_docker.sh ]; then
        echo "Run in a separate terminal:"
        echo "  ~/start_jetracer_docker.sh"
        echo ""
        echo "Press Enter when Docker is running..."
        read -r
    else
        echo "❌ Docker start script not found at ~/start_jetracer_docker.sh"
        exit 1
    fi
fi

echo ""

# Step 4: Run the system
echo "Step 4: Starting JetRacer control system..."
echo "----------------------------------------"
echo ""
echo "Select mode:"
echo "  1. Manual (Keyboard)"
echo "  2. Autonomous (AI)"
echo "  3. Exit"
echo ""
read -p "Choice (1-3): " mode_choice

case $mode_choice in
    1)
        echo ""
        echo "Starting Manual Mode..."
        python3 main.py --mode manual
        ;;
    2)
        echo ""
        echo "Starting Autonomous Mode..."
        echo ""
        echo "⚠️  SAFETY REMINDER:"
        echo "  - Test in open area first"
        echo "  - Keep emergency stop ready (Ctrl+C)"
        echo "  - Monitor robot behavior"
        echo ""
        read -p "Ready to start autonomous mode? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python3 main.py --mode autonomous
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Done! 🎉"

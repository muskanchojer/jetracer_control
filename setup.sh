#!/bin/bash
# Setup script for JetRacer Autonomous Control System
# For Ubuntu 22.04 (Jetson Orin Nano)

set -e  # Exit on error

echo "=================================================="
echo "  JetRacer Autonomous Control System - Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo -e "${GREEN}✓${NC} Detected NVIDIA Jetson platform"
    IS_JETSON=true
else
    echo -e "${YELLOW}!${NC} Not running on Jetson - some features may not work"
    IS_JETSON=false
fi

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Docker is installed
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    docker --version
else
    echo -e "${RED}✗${NC} Docker is not installed"
    echo "Please install Docker first"
    exit 1
fi

# Check if Docker container exists
echo ""
echo "Checking for jetracer_ros container..."
if docker ps -a --format '{{.Names}}' | grep -q "^jetracer_ros$"; then
    echo -e "${GREEN}✓${NC} jetracer_ros container exists"
else
    echo -e "${YELLOW}!${NC} jetracer_ros container not found"
    echo "Make sure to create it with ~/start_jetracer_docker.sh"
fi

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    v4l-utils

echo -e "${GREEN}✓${NC} System dependencies installed"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}!${NC} Virtual environment already exists"
    echo "Do you want to recreate it? (y/n)"
    read -r recreate_venv
    if [ "$recreate_venv" = "y" ] || [ "$recreate_venv" = "Y" ]; then
        rm -rf .venv
        python3 -m venv .venv --system-site-packages
        echo -e "${GREEN}✓${NC} Virtual environment recreated"
    fi
else
    python3 -m venv .venv --system-site-packages
    echo -e "${GREEN}✓${NC} Virtual environment created with system site packages"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo ""
echo "Installing Python packages..."
pip install -r requirements.txt

echo -e "${GREEN}✓${NC} Python packages installed in virtual environment"

# Download YOLOv8 model if not present
echo ""
echo "Checking for YOLOv8 model..."
MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/yolov8n.pt" ]; then
    echo "Downloading YOLOv8n model..."
    python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model downloaded')"
    mv yolov8n.pt "$MODEL_DIR/"
    echo -e "${GREEN}✓${NC} YOLOv8 model downloaded"
else
    echo -e "${GREEN}✓${NC} YOLOv8 model already exists"
fi

# Optional: Install tinygrad
# echo ""
# echo "Do you want to install tinygrad for lightweight inference? (y/n)"
# read -r install_tinygrad

# if [ "$install_tinygrad" = "y" ] || [ "$install_tinygrad" = "Y" ]; then
#     echo "Installing tinygrad..."
#     if [ -d "tinygrad" ]; then
#         cd tinygrad && git pull && cd ..
#     else
#         git clone https://github.com/tinygrad/tinygrad.git
#     fi
#     cd tinygrad
#     pip install -e .
#     cd ..
#     echo -e "${GREEN}✓${NC} tinygrad installed"
    
#     # Download YOLOv8 safetensors
#     echo "Downloading YOLOv8 safetensors for tinygrad..."
#     if [ ! -f "$MODEL_DIR/yolov8n.safetensors" ]; then
#         wget -O "$MODEL_DIR/yolov8n.safetensors" \
#             "https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8n.safetensors"
#         echo -e "${GREEN}✓${NC} YOLOv8 safetensors downloaded"
#     fi
# else
#     echo "Skipping tinygrad installation"
#     echo "Note: Autonomous mode will use PyTorch/ultralytics backend"
# fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x main.py
chmod +x main_old.py
chmod +x scripts/*.py
chmod +x auto/*.py

echo -e "${GREEN}✓${NC} Scripts made executable"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p data/trajectories

echo -e "${GREEN}✓${NC} Directories created"

# Summary
echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Start Docker container: ~/start_jetracer_docker.sh"
echo "  3. Run the system: python3 main.py"
echo ""
echo "Modes available:"
echo "  - Manual mode: python3 main.py --mode manual"
echo "  - Autonomous mode: python3 main.py --mode autonomous"
echo ""
echo -e "${YELLOW}Important:${NC} Always activate the venv before running:"
echo "  source .venv/bin/activate"
echo ""
echo "For autonomous mode with trained model:"
echo "  - Place tinynet.safetensors in commabot/dollop-main/"
echo "  - Or train your own: cd commabot/dollop-main && ./train.py"
echo ""
echo -e "${GREEN}Happy driving! 🚗${NC}"

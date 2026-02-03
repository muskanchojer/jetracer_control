# JetRacer Autonomous System - Virtual Environment Guide

## Why Use a Virtual Environment?

✅ **Isolates project dependencies** from system packages  
✅ **Prevents version conflicts** with CUDA and PyTorch  
✅ **Easy to recreate** if something breaks  
✅ **Clean development environment**  

## Setup (First Time)

```bash
# Run the setup script (creates venv automatically)
./setup.sh
```

This creates a `.venv` folder with `--system-site-packages` flag, which:
- Gives access to Jetson's CUDA libraries
- Uses system OpenCV (better compatibility)
- Installs project packages in isolation

## Daily Usage

### Option 1: Source the activate script
```bash
source activate.sh
```

### Option 2: Manual activation
```bash
source .venv/bin/activate
```

### Check if activated
```bash
which python3
# Should show: ~/jetracer_control/.venv/bin/python3
```

### Run your code
```bash
python3 main.py
```

### Deactivate when done
```bash
deactivate
```

## Working With Packages

### Install additional packages
```bash
source .venv/bin/activate
pip install package-name
```

### Update requirements.txt
```bash
pip freeze > requirements.txt
```

### Reinstall from requirements
```bash
pip install -r requirements.txt
```

## Tinygrad Installation

If you want to use tinygrad (lightweight ML framework):

```bash
source .venv/bin/activate
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
pip install -e .
cd ..
```

Then set in `config/autonomous_config.yaml`:
```yaml
inference:
  use_tinygrad: true
```

## Troubleshooting

### "No module named 'X'" error
```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### CUDA/PyTorch issues
```bash
# Check PyTorch can see CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, you may need Jetson-specific PyTorch
# (system-site-packages should handle this)
```

### OpenCV issues
```bash
# Check if accessible
python3 -c "import cv2; print(cv2.__version__)"

# If not found, install in venv:
pip install opencv-python
```

### Delete and recreate venv
```bash
deactivate  # if currently activated
rm -rf .venv
./setup.sh  # recreate
```

## VS Code Integration

If using VS Code, set Python interpreter:

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose: `.venv/bin/python3`

## Docker vs Virtual Environment

**Important distinction:**

- **Docker container** (`jetracer_ros`): Runs ROS Noetic, receives motor commands
- **Virtual environment** (`.venv`): Runs Python AI code on host system

```
Host System (.venv):          Docker Container:
├── YOLOv8 inference      →   ├── ROS Noetic
├── TinyNet policy        →   ├── Motor control
└── Commands sent ────────→   └── Hardware drivers
```

The AI runs on host (in venv), sends commands to Docker.

## Quick Reference

```bash
# Activate
source activate.sh

# Run manual mode
python3 main.py --mode manual

# Run autonomous mode
python3 main.py --mode autonomous

# Test system
python3 preflight_check.py

# Deactivate
deactivate
```

## Best Practices

✅ **Always activate** before running Python scripts  
✅ **Update requirements.txt** when adding packages  
✅ **Use system packages** for CUDA/OpenCV when possible  
✅ **Keep venv in .gitignore** (don't commit it)  

## Summary

**For everyday use:**
```bash
cd ~/jetracer_control
source activate.sh  # or: source .venv/bin/activate
python3 main.py
deactivate  # when done
```

That's it! The virtual environment keeps everything organized and prevents conflicts.

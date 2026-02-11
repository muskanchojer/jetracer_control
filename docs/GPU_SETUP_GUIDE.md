# GPU Setup Guide for Jetson Orin Nano

## Current Status
- ✅ CUDA toolkit installed (12.6)
- ❌ PyTorch is CPU-only version (2.9.1+cpu)
- ❌ No GPU acceleration available

## Install PyTorch with CUDA for Jetson

### Option 1: Using NVIDIA's Pre-built Wheels (Recommended)

1. **Uninstall CPU-only PyTorch:**
```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
```

2. **Install PyTorch for Jetson (JetPack 6.x compatible):**
```bash
# For JetPack 6.x (CUDA 12.x)
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR if the above doesn't work, use NVIDIA's pre-built wheels:
# Download from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Example for PyTorch 2.1.0:
wget https://nvidia.box.com/shared/static/[hash].whl -O torch-2.1.0-cp311-cp311-linux_aarch64.whl
pip install torch-2.1.0-cp311-cp311-linux_aarch64.whl
```

3. **Verify GPU is available:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Option 2: Build from Source (Advanced)

Follow NVIDIA's guide: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

## Performance Optimizations

### 1. Enable TensorRT for YOLOv8 (Fastest)
```bash
# Install TensorRT (should be pre-installed on Jetson)
pip install nvidia-tensorrt

# Export YOLOv8 to TensorRT engine
python -c "from ultralytics import YOLO; model = YOLO('models/yolov8n.pt'); model.export(format='engine', device=0)"
```

Then update [config/autonomous_config.yaml](config/autonomous_config.yaml):
```yaml
models:
  yolov8_weights: "models/yolov8n.engine"  # Use TensorRT engine
```

### 2. Use Mixed Precision (FP16)
Already enabled in [auto/vision.py](auto/vision.py) when GPU is detected.

### 3. Optimize Inference Settings
In [config/autonomous_config.yaml](config/autonomous_config.yaml):
```yaml
vision:
  use_gpu: true
  device: "cuda"
  
inference:
  use_tinygrad: false  # PyTorch is faster with GPU
  fps_target: 15       # Can increase with GPU
```

## Expected Performance

| Backend | Device | FPS | Notes |
|---------|--------|-----|-------|
| tinygrad | CPU | 1.6 | Current (very slow) |
| PyTorch | CPU | 3-5 | Better optimization |
| PyTorch | GPU | 10-15 | Good performance |
| TensorRT | GPU | 20-30+ | Best performance |

## Troubleshooting

### Check CUDA Installation
```bash
nvcc --version              # Should show CUDA 12.6
nvidia-smi                  # Should show GPU info
```

### Check PyTorch Build
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
```

### Test GPU Inference
```bash
python << EOF
import torch
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    model = YOLO('models/yolov8n.pt')
    model.to('cuda')
    results = model('path/to/image.jpg', device=0)
    print("GPU inference successful!")
EOF
```

## Current Configuration Changes

**Fixed Issues:**
1. ✅ ROS infrastructure now starts properly (docker exec -d)
2. ✅ Switched to PyTorch backend (faster than tinygrad)
3. 📝 Need to install PyTorch with CUDA support

**After GPU Setup, Expected Improvements:**
- FPS: 1.6 → 10-15 FPS
- Latency: ~650ms → ~70ms per frame
- Robot responsiveness: Much better real-time control

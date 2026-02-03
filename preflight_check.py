#!/usr/bin/env python3
"""
Pre-flight Check - Validate all components before autonomous run
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check all required imports"""
    print("Checking imports...")
    errors = []
    
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError as e:
        errors.append(f"yaml: {e}")
    
    try:
        import cv2
        print("  ✓ opencv (cv2)")
    except ImportError as e:
        errors.append(f"opencv: {e}")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    try:
        from ultralytics import YOLO
        print("  ✓ ultralytics (YOLO)")
    except ImportError as e:
        errors.append(f"ultralytics: {e}")
    
    try:
        import tinygrad
        print("  ✓ tinygrad")
    except ImportError as e:
        print("  ⚠ tinygrad (optional, but needed for trained model)")
    
    try:
        import safetensors
        print("  ✓ safetensors")
    except ImportError as e:
        errors.append(f"safetensors: {e}")
    
    if errors:
        print("\n❌ Import errors found:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✅ All imports OK\n")
    return True


def check_configs():
    """Check configuration files"""
    print("Checking configuration files...")
    errors = []
    
    config_dir = Path(__file__).parent / "config"
    
    # Check robot_config.yaml
    robot_config = config_dir / "robot_config.yaml"
    if not robot_config.exists():
        errors.append(f"Missing: {robot_config}")
    else:
        print(f"  ✓ {robot_config.name}")
        try:
            import yaml
            with open(robot_config) as f:
                config = yaml.safe_load(f)
                # Check key fields
                if 'hardware' not in config:
                    errors.append("robot_config.yaml missing 'hardware' section")
        except Exception as e:
            errors.append(f"Error loading robot_config.yaml: {e}")
    
    # Check autonomous_config.yaml
    auto_config = config_dir / "autonomous_config.yaml"
    if not auto_config.exists():
        errors.append(f"Missing: {auto_config}")
    else:
        print(f"  ✓ {auto_config.name}")
        try:
            import yaml
            with open(auto_config) as f:
                config = yaml.safe_load(f)
                # Check key fields
                required = ['models', 'vision', 'control_mapping', 'safety', 'inference']
                for key in required:
                    if key not in config:
                        errors.append(f"autonomous_config.yaml missing '{key}' section")
                
                # Check inference section
                if 'inference' in config:
                    if 'prediction_rate' not in config['inference']:
                        errors.append("autonomous_config.yaml missing 'inference.prediction_rate'")
        except Exception as e:
            errors.append(f"Error loading autonomous_config.yaml: {e}")
    
    if errors:
        print("\n❌ Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✅ All configs OK\n")
    return True


def check_modules():
    """Check project modules can be imported"""
    print("Checking project modules...")
    errors = []
    
    try:
        from auto.vision import VisionProcessor, CameraStream
        print("  ✓ auto.vision")
    except Exception as e:
        errors.append(f"auto.vision: {e}")
    
    try:
        from auto.policy import PolicyNetwork, PolicyEvaluator
        print("  ✓ auto.policy")
    except Exception as e:
        errors.append(f"auto.policy: {e}")
    
    try:
        from auto.control_adapter import ControlAdapter, SafetyMonitor
        print("  ✓ auto.control_adapter")
    except Exception as e:
        errors.append(f"auto.control_adapter: {e}")
    
    try:
        from scripts.autonomous_control import AutonomousController
        print("  ✓ scripts.autonomous_control")
    except Exception as e:
        errors.append(f"scripts.autonomous_control: {e}")
    
    if errors:
        print("\n❌ Module errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✅ All modules OK\n")
    return True


def check_hardware():
    """Check hardware availability"""
    print("Checking hardware...")
    warnings = []
    
    # Check camera
    camera = "/dev/video0"
    if os.path.exists(camera):
        print(f"  ✓ Camera at {camera}")
    else:
        warnings.append(f"Camera not found at {camera}")
    
    # Check robot serial
    robot = "/dev/ttyACM0"
    if os.path.exists(robot):
        print(f"  ✓ Robot at {robot}")
    else:
        warnings.append(f"Robot not found at {robot}")
    
    # Check Docker
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=jetracer_ros", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        if "jetracer_ros" in result.stdout:
            print("  ✓ Docker container 'jetracer_ros' running")
        else:
            warnings.append("Docker container 'jetracer_ros' not running")
    except Exception as e:
        warnings.append(f"Cannot check Docker: {e}")
    
    if warnings:
        print("\n⚠ Hardware warnings:")
        for warn in warnings:
            print(f"  - {warn}")
        print("(These may be OK if testing without hardware)\n")
    else:
        print("✅ All hardware OK\n")
    
    return True


def check_trained_model():
    """Check if trained model exists"""
    print("Checking trained model...")
    
    model_path = Path(__file__).parent / "commabot" / "dollop-main" / "tinynet.safetensors"
    
    if model_path.exists():
        size = model_path.stat().st_size
        print(f"  ✓ Trained model found: {model_path}")
        print(f"    Size: {size:,} bytes")
        print("✅ Model OK\n")
        return True
    else:
        print(f"  ⚠ Trained model not found at: {model_path}")
        print("    System will use untrained model (random predictions)\n")
        return True


def main():
    """Run all checks"""
    print("="*60)
    print("JetRacer Autonomous Control - Pre-flight Check")
    print("="*60)
    print()
    
    all_ok = True
    
    all_ok &= check_imports()
    all_ok &= check_configs()
    all_ok &= check_modules()
    all_ok &= check_hardware()
    all_ok &= check_trained_model()
    
    print("="*60)
    if all_ok:
        print("✅ PRE-FLIGHT CHECK PASSED")
        print("="*60)
        print("\nYou can now run:")
        print("  python main.py --mode autonomous")
        print()
        return 0
    else:
        print("❌ PRE-FLIGHT CHECK FAILED")
        print("="*60)
        print("\nPlease fix the errors above before running autonomous mode.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Vision module - YOLOv8 feature extraction for autonomous navigation
Supports both PyTorch (ultralytics) and tinygrad implementations
"""

import os
import sys
import numpy as np
import cv2
import logging
from pathlib import Path

class VisionProcessor:
    """Handles camera input and YOLOv8 feature extraction"""
    
    def __init__(self, config, use_tinygrad=False):
        self.config = config
        self.use_tinygrad = use_tinygrad
        self.logger = logging.getLogger(__name__)
        
        self.width = config['vision']['camera_width']
        self.height = config['vision']['camera_height']
        
        if use_tinygrad:
            self._init_tinygrad()
        else:
            self._init_pytorch()
    
    def _init_pytorch(self):
        """Initialize PyTorch-based YOLOv8 (easier, more compatible)"""
        try:
            import torch
            from ultralytics import YOLO
            
            # Check GPU availability
            use_gpu = self.config.get('vision', {}).get('use_gpu', False)
            device = self.config.get('vision', {}).get('device', 'cpu')
            
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                if use_gpu:
                    self.logger.warning("GPU requested but not available, using CPU")
            
            self.logger.info(f"Initializing YOLOv8 with PyTorch/ultralytics on {self.device}")
            
            model_path = self.config['models']['yolov8_weights']
            if not os.path.exists(model_path):
                self.logger.warning(f"YOLOv8 weights not found at {model_path}, downloading...")
                self.yolo = YOLO('yolov8n.pt')  # Auto-download
            else:
                self.yolo = YOLO(model_path)
            
            # Move model to GPU if available
            self.yolo.to(self.device)
            
            # Enable FP16 for faster inference on GPU
            if self.device == 'cuda':
                self.use_fp16 = True
                self.logger.info("Enabled FP16 mixed precision for GPU")
            else:
                self.use_fp16 = False
            
            self.logger.info(f"YOLOv8 loaded successfully on {self.device}")
            
        except ImportError:
            self.logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
    
    def _init_tinygrad(self):
        """Initialize tinygrad-based YOLOv8 (lightweight, for edge devices)"""
        try:
            from tinygrad.tensor import Tensor
            from tinygrad.nn.state import load_state_dict, safe_load
            
            # Add dollop code to path
            dollop_path = str(Path(__file__).parent.parent / "commabot" / "dollop-main")
            if dollop_path not in sys.path:
                sys.path.insert(0, dollop_path)
            
            from yolov8 import YOLOv8, get_variant_multiples
            
            self.logger.info("Initializing YOLOv8 with tinygrad")
            
            # Load YOLOv8 architecture
            yolo_variant = "n"
            depth, width, ratio = get_variant_multiples(yolo_variant)
            self.yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)
            
            # Load weights
            weights_path = self.config['models']['yolov8_safetensors']
            if not os.path.exists(weights_path):
                self.logger.error(f"YOLOv8 safetensors not found at {weights_path}")
                self.logger.info("Download from: https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8n.safetensors")
                raise FileNotFoundError(f"Missing {weights_path}")
            
            state_dict = safe_load(weights_path)
            load_state_dict(self.yolo_infer, state_dict)
            
            self.logger.info("YOLOv8 tinygrad model loaded")
            
        except ImportError as e:
            self.logger.error(f"tinygrad import failed: {e}")
            self.logger.info("Install tinygrad from: https://github.com/tinygrad/tinygrad")
            raise
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame to 640x480 RGB"""
        if frame is None:
            return None
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected dimensions
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def extract_features_pytorch(self, frame):
        """Extract YOLOv8 FPN features using PyTorch - simplified for tinygrad compatibility"""
        # Note: This is a fallback. The trained model expects tinygrad YOLOv8 FPN features.
        # For proper use, keep use_tinygrad=true in config
        try:
            import torch
            
            # Since the trained model expects 256-channel features from YOLOv8 FPN,
            # and extracting those from PyTorch YOLO is complex, we just return
            # a warning and simplified features
            self.logger.warning("PyTorch backend may not match trained model expectations")
            self.logger.warning("Recommend using tinygrad backend (use_tinygrad: true in config)")
            
            # Return normalized frame as placeholder
            frame_normalized = frame.astype(np.float32) / 255.0
            return frame_normalized
            
        except Exception as e:
            self.logger.error(f"Error in PyTorch feature extraction: {e}")
            return None
    
    def extract_features_tinygrad(self, frame):
        """Extract YOLOv8 features using tinygrad"""
        from tinygrad.tensor import Tensor
        
        # Convert to tensor (batch, height, width, channels)
        imgs = Tensor(frame.reshape(1, self.height, self.width, 3))
        
        # Run through YOLOv8 feature pyramid network
        x = self.yolo_infer.net(imgs.permute(0, 3, 1, 2).float() / 255)
        x = self.yolo_infer.fpn(*x)
        
        # Return the last feature map (x[2])
        features = x[2].realize()
        return features
    
    def process_frame(self, frame):
        """Main processing pipeline: preprocess + extract features"""
        # Preprocess
        frame = self.preprocess_frame(frame)
        if frame is None:
            return None
        
        # Extract features based on backend
        if self.use_tinygrad:
            return self.extract_features_tinygrad(frame)
        else:
            return self.extract_features_pytorch(frame)
    
    def get_camera_frame(self, camera_device='/dev/video0'):
        """Capture a single frame from camera"""
        cap = cv2.VideoCapture(camera_device)
        if not cap.isOpened():
            self.logger.error(f"Cannot open camera: {camera_device}")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            self.logger.error("Failed to capture frame")
            return None
        
        return frame


class CameraStream:
    """Continuous camera stream for real-time processing"""
    
    def __init__(self, device='/dev/video0', width=640, height=480):
        self.device = device
        self.width = width
        self.height = height
        self.cap = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start camera stream"""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open camera: {self.device}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.logger.info(f"Camera stream started: {self.device}")
        return True
    
    def read(self):
        """Read a frame from the stream"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def stop(self):
        """Stop camera stream"""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera stream stopped")

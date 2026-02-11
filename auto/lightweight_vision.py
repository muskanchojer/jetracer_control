#!/usr/bin/env python3
"""
Lightweight Vision Module - Fast alternatives to YOLOv8 for Jetson Orin Nano
Options:
1. DirectPolicyNet - End-to-end CNN, no feature extractor needed
2. MobileNetV3 - Lightweight backbone with adapter to 256 channels
3. EfficientNet-Lite - Optimized for edge devices

All options maintain compatibility with existing TinyNet policy.
"""

import os
import sys
import time
import numpy as np
import cv2
import logging
import threading
import queue
from pathlib import Path


class DirectPolicyNet:
    """
    Ultra-lightweight end-to-end policy network.
    Takes raw images directly, no separate feature extractor.
    ~10x faster than YOLOv8 + TinyNet pipeline.
    
    Architecture: Raw Image -> Conv layers -> Action
    Must be trained from scratch on your dataset.
    """
    
    def __init__(self, config, use_tinygrad=True):
        self.config = config
        self.use_tinygrad = use_tinygrad
        self.logger = logging.getLogger(__name__)
        self.width = config['vision']['camera_width']
        self.height = config['vision']['camera_height']
        self.class_names = ["left", "straight", "right", "reverse"]
        
        if use_tinygrad:
            self._init_tinygrad()
        else:
            self._init_pytorch()
    
    def _init_tinygrad(self):
        """Initialize tinygrad direct policy"""
        from tinygrad.tensor import Tensor
        from tinygrad.nn import Conv2d, Linear
        from tinygrad.nn.state import load_state_dict, safe_load
        
        self.logger.info("Initializing DirectPolicyNet with tinygrad")
        
        # Lightweight CNN - processes 160x120 resized images
        # Much faster than 640x480 through YOLOv8
        class DirectNet:
            def __init__(self):
                # Input: 3 channels (RGB), 160x120
                self.c1 = Conv2d(3, 16, 5, stride=2, padding=2)   # -> 16x80x60
                self.c2 = Conv2d(16, 32, 3, stride=2, padding=1)  # -> 32x40x30
                self.c3 = Conv2d(32, 64, 3, stride=2, padding=1)  # -> 64x20x15
                self.c4 = Conv2d(64, 64, 3, stride=2, padding=1)  # -> 64x10x7
                self.fc1 = Linear(64 * 10 * 7, 64)
                self.fc2 = Linear(64, 4)
            
            def __call__(self, x):
                # x: (B, 3, 120, 160) normalized to 0-1
                x = self.c1(x).relu()
                x = self.c2(x).relu()
                x = self.c3(x).relu()
                x = self.c4(x).relu()
                x = x.reshape(x.shape[0], -1)
                x = self.fc1(x).relu().dropout(0.3)
                x = self.fc2(x)
                return x.log_softmax()
        
        self.net = DirectNet()
        
        # Try to load weights
        weights_path = self.config.get('models', {}).get('direct_policy_model', 
                                                          'models/direct_policy.safetensors')
        if os.path.exists(weights_path):
            state_dict = safe_load(weights_path)
            load_state_dict(self.net, state_dict)
            self.logger.info(f"DirectPolicyNet loaded from {weights_path}")
        else:
            self.logger.warning(f"No weights at {weights_path} - using random initialization")
            self.logger.warning("Train with: python train_direct_policy.py")
        
        Tensor.training = False
        Tensor.no_grad = True
        
    def _init_pytorch(self):
        """Initialize PyTorch direct policy"""
        import torch
        import torch.nn as nn
        
        self.logger.info("Initializing DirectPolicyNet with PyTorch")
        
        # Check GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        class DirectNetPyTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 5, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(64 * 10 * 7, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, 4),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        self.net = DirectNetPyTorch().to(self.device)
        self.net.eval()
        
        weights_path = self.config.get('models', {}).get('direct_policy_model',
                                                          'models/direct_policy.pt')
        if os.path.exists(weights_path):
            self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.logger.info(f"DirectPolicyNet loaded from {weights_path}")
        else:
            self.logger.warning(f"No weights at {weights_path}")
    
    def preprocess(self, frame):
        """Preprocess frame: resize to 160x120, normalize"""
        # Resize to small resolution
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        if len(small.shape) == 3 and small.shape[2] == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        small = small.astype(np.float32) / 255.0
        
        return small
    
    def predict(self, frame):
        """Predict action from raw camera frame"""
        # Preprocess
        processed = self.preprocess(frame)
        
        if self.use_tinygrad:
            return self._predict_tinygrad(processed)
        else:
            return self._predict_pytorch(processed)
    
    def _predict_tinygrad(self, processed):
        """Tinygrad prediction"""
        from tinygrad.tensor import Tensor
        
        # (H, W, C) -> (1, C, H, W)
        x = Tensor(processed.transpose(2, 0, 1).reshape(1, 3, 120, 160))
        
        logits = self.net(x)
        probs = logits.exp().realize()
        probs_np = probs.numpy()[0]
        
        pred_class = np.argmax(probs_np)
        confidence = probs_np[pred_class]
        action = self.class_names[pred_class]
        
        return action, confidence, probs_np
    
    def _predict_pytorch(self, processed):
        """PyTorch prediction"""
        import torch
        
        # (H, W, C) -> (1, C, H, W)
        x = torch.from_numpy(processed.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.net(x)
            probs = torch.softmax(logits, dim=-1)
        
        probs_np = probs.cpu().numpy()[0]
        pred_class = np.argmax(probs_np)
        confidence = probs_np[pred_class]
        action = self.class_names[pred_class]
        
        return action, confidence, probs_np


class MobileNetAdapter:
    """
    Uses MobileNetV3-Small as feature extractor with adapter to 256 channels.
    Compatible with existing TinyNet policy (expects 256 channels).
    
    Optimizations for Jetson Orin Nano:
    - FP16 inference
    - TorchScript JIT compilation
    - Reduced input resolution (160x160)
    - CUDA memory optimization
    - Optional TensorRT conversion
    """
    
    def __init__(self, config, use_tinygrad=False):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.width = config['vision']['camera_width']
        self.height = config['vision']['camera_height']
        
        # Optimization settings from config
        opt_config = config.get('vision', {}).get('optimization', {})
        self.input_size = opt_config.get('mobilenet_input_size', 160)  # Smaller = faster
        self.use_tensorrt = opt_config.get('use_tensorrt', False)
        self.use_jit = opt_config.get('use_jit', True)
        
        # MobileNet works better with PyTorch (pretrained weights)
        self._init_pytorch()
    
    def _init_pytorch(self):
        """Initialize MobileNetV3-Small + adapter with optimizations"""
        import torch
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Initializing MobileNetV3-Small on {self.device}")
        self.logger.info(f"  Input size: {self.input_size}x{self.input_size}")
        
        # Load pretrained MobileNetV3-Small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Create combined model for better optimization
        class MobileNetFeatureExtractor(nn.Module):
            def __init__(self, features, adapter):
                super().__init__()
                self.features = features
                self.adapter = adapter
            
            def forward(self, x):
                x = self.features(x)
                x = self.adapter(x)
                return x
        
        # Remove classifier, keep features + adapter
        features = backbone.features
        adapter = nn.Conv2d(576, 256, kernel_size=1, bias=False)
        
        self.model = MobileNetFeatureExtractor(features, adapter)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Enable FP16 for faster inference on GPU
        if self.device == 'cuda':
            self.model = self.model.half()
            self.use_fp16 = True
            self.logger.info("  FP16 enabled")
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.use_fp16 = False
        
        # JIT compile for faster execution
        if self.use_jit and not self.use_tensorrt:
            try:
                dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
                if self.use_fp16:
                    dummy_input = dummy_input.half()
                
                self.model = torch.jit.trace(self.model, dummy_input)
                self.model = torch.jit.freeze(self.model)
                
                # Warmup runs for JIT optimization
                for _ in range(3):
                    _ = self.model(dummy_input)
                
                self.logger.info("  TorchScript JIT compiled & frozen")
            except Exception as e:
                self.logger.warning(f"  JIT compilation failed: {e}")
        
        # TensorRT optimization (best performance on Jetson)
        if self.use_tensorrt and self.device == 'cuda':
            self._convert_to_tensorrt()
        
        # Pre-allocate tensors for zero-copy preprocessing
        self._setup_preprocessing_buffers()
        
        self.logger.info("MobileNetV3-Small initialized with optimizations")
    
    def _convert_to_tensorrt(self):
        """Convert model to TensorRT for maximum Jetson performance"""
        try:
            import torch_tensorrt
            
            self.logger.info("  Converting to TensorRT...")
            
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            if self.use_fp16:
                dummy_input = dummy_input.half()
            
            # TensorRT compilation
            self.model = torch_tensorrt.compile(
                self.model,
                inputs=[torch_tensorrt.Input(
                    shape=[1, 3, self.input_size, self.input_size],
                    dtype=torch.half if self.use_fp16 else torch.float
                )],
                enabled_precisions={torch.half} if self.use_fp16 else {torch.float},
                truncate_long_and_double=True
            )
            self.logger.info("  TensorRT conversion successful!")
            
        except ImportError:
            self.logger.warning("  torch_tensorrt not installed, skipping TensorRT")
            self.logger.info("  Install with: pip install torch-tensorrt")
        except Exception as e:
            self.logger.warning(f"  TensorRT conversion failed: {e}")
    
    def _setup_preprocessing_buffers(self):
        """Pre-allocate buffers for faster preprocessing"""
        import torch
        
        # Pre-compute normalization tensors on GPU
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        if self.use_fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        
        # Pre-allocate input tensor
        self._input_buffer = torch.empty(
            1, 3, self.input_size, self.input_size,
            device=self.device,
            dtype=torch.half if self.use_fp16 else torch.float
        )
    
    def preprocess_frame(self, frame):
        """Optimized preprocessing with minimal CPU-GPU transfer"""
        import torch
        
        # Resize to smaller input (160x160 default, much faster than 224x224)
        resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize on GPU (faster than CPU normalization)
        rgb = rgb.astype(np.float32) / 255.0
        
        # (H, W, C) -> (1, C, H, W) and copy to pre-allocated buffer
        self._input_buffer.copy_(
            torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(
                device=self.device,
                dtype=torch.half if self.use_fp16 else torch.float
            )
        )
        
        # GPU-side normalization
        normalized = (self._input_buffer - self.mean) / self.std
        
        return normalized
    
    def process_frame(self, frame):
        """Extract features compatible with TinyNet"""
        import torch
        
        # Preprocess
        x = self.preprocess_frame(frame)
        
        # Extract features using combined optimized model
        with torch.no_grad():
            features = self.model(x)  # (1, 256, H, W)
        
        return features


class LightweightVisionProcessor:
    """
    Drop-in replacement for VisionProcessor with lightweight options.
    Maintains same interface for compatibility.
    """
    
    def __init__(self, config, mode='direct', use_tinygrad=True):
        """
        Args:
            config: Configuration dict
            mode: 'direct' (fastest), 'mobilenet' (compatible with TinyNet), 'yolov8' (original)
            use_tinygrad: Use tinygrad backend
        """
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        self.width = config['vision']['camera_width']
        self.height = config['vision']['camera_height']
        
        if mode == 'direct':
            self.logger.info("Using DirectPolicyNet (fastest, end-to-end)")
            self.processor = DirectPolicyNet(config, use_tinygrad=use_tinygrad)
            self.is_end_to_end = True
        elif mode == 'mobilenet':
            self.logger.info("Using MobileNetV3 + TinyNet (TinyNet compatible)")
            self.processor = MobileNetAdapter(config, use_tinygrad=False)
            self.is_end_to_end = False
        else:
            self.logger.info("Using YOLOv8 (original)")
            from auto.vision import VisionProcessor
            self.processor = VisionProcessor(config, use_tinygrad=use_tinygrad)
            self.is_end_to_end = False
    
    def preprocess_frame(self, frame):
        """Preprocess frame"""
        if frame is None:
            return None
        
        if hasattr(self.processor, 'preprocess_frame'):
            return self.processor.preprocess_frame(frame)
        elif hasattr(self.processor, 'preprocess'):
            return self.processor.preprocess(frame)
        return frame
    
    def process_frame(self, frame):
        """Process frame and return features or prediction"""
        if self.mode == 'direct':
            # DirectPolicyNet returns action directly, but we need to wrap it
            # For compatibility, return None (caller should use predict directly)
            return None
        else:
            return self.processor.process_frame(frame)
    
    def predict(self, frame_or_features):
        """
        Predict action from frame or features.
        For 'direct' mode: pass raw frame
        For other modes: pass extracted features
        """
        if self.mode == 'direct':
            return self.processor.predict(frame_or_features)
        else:
            # Features mode - caller handles this
            raise ValueError("For non-direct modes, use separate policy network")


class AsyncLightweightVision:
    """
    Async wrapper for lightweight vision processing.
    Same interface as AsyncVisionProcessor.
    """
    
    def __init__(self, lightweight_processor, skip_frames=2):
        self.processor = lightweight_processor
        self.skip_frames = max(1, skip_frames)
        self.logger = logging.getLogger(__name__)
        
        self._lock = threading.Lock()
        self._latest_result = None
        self._result_timestamp = 0
        self._frame_count = 0
        self._processed_count = 0
        
        self._input_queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread = None
        
        self._fps = 0.0
        self._processing_time = 0.0
        self._last_fps_time = 0
        self._fps_count = 0
    
    def start(self):
        """Start processing thread"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"AsyncLightweightVision started (mode={self.processor.mode})")
    
    def stop(self):
        """Stop processing thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def submit_frame(self, frame):
        """Submit frame for processing"""
        self._frame_count += 1
        
        if self._frame_count % self.skip_frames != 0:
            return
        
        try:
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
            self._input_queue.put_nowait((frame.copy(), self._frame_count))
        except queue.Full:
            pass
    
    def _process_loop(self):
        """Background processing"""
        while self._running:
            try:
                frame, frame_id = self._input_queue.get(timeout=0.1)
                start_time = time.time()
                
                if self.processor.is_end_to_end:
                    # Direct prediction from frame
                    action, confidence, probs = self.processor.predict(frame)
                    
                    result = {
                        'action': action,
                        'confidence': confidence,
                        'probs': probs,
                        'frame_id': frame_id,
                        'timestamp': time.time()
                    }
                    
                    with self._lock:
                        self._latest_result = result
                        self._result_timestamp = result['timestamp']
                        self._processed_count += 1
                
                self._processing_time = time.time() - start_time
                self._fps_count += 1
                
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._fps = self._fps_count / (now - self._last_fps_time)
                    self._fps_count = 0
                    self._last_fps_time = now
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
    
    def get_result(self, max_age=0.5):
        """Get latest result"""
        with self._lock:
            if self._latest_result is None:
                return None
            if time.time() - self._result_timestamp > max_age:
                return None
            return self._latest_result.copy()
    
    def get_stats(self):
        """Get processing stats"""
        return {
            'fps': self._fps,
            'processing_time_ms': self._processing_time * 1000,
            'frames_submitted': self._frame_count,
            'frames_processed': self._processed_count,
            'skip_frames': self.skip_frames,
            'mode': self.processor.mode
        }

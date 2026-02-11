#!/usr/bin/env python3
"""
Vision module - YOLOv8 feature extraction for autonomous navigation
Supports both PyTorch (ultralytics) and tinygrad implementations
Optimized for Jetson Orin Nano with async processing
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
            # Force GPU device for tinygrad
            import os
            os.environ['CUDA'] = '1'  # Enable CUDA
            os.environ['GPU'] = '1'   # Enable GPU
            
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


class AsyncVisionProcessor:
    """
    Asynchronous vision processor that runs in a separate thread
    Prevents vision processing from blocking the main control loop
    """
    
    def __init__(self, vision_processor, policy_network, skip_frames=2):
        """
        Args:
            vision_processor: VisionProcessor instance
            policy_network: PolicyNetwork instance  
            skip_frames: Process every Nth frame (1=all, 2=every other, etc.)
        """
        self.vision = vision_processor
        self.policy = policy_network
        self.skip_frames = max(1, skip_frames)
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe result storage
        self._lock = threading.Lock()
        self._latest_result = None
        self._result_timestamp = 0
        self._frame_count = 0
        self._processed_count = 0
        
        # Processing queue and thread
        self._input_queue = queue.Queue(maxsize=1)  # Only keep latest frame
        self._running = False
        self._thread = None
        
        # Performance tracking
        self._processing_time = 0.0
        self._fps = 0.0
        self._last_fps_time = 0
        self._fps_count = 0
    
    def start(self):
        """Start the async processing thread"""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Async vision processor started (skip_frames={self.skip_frames})")
    
    def stop(self):
        """Stop the async processing thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("Async vision processor stopped")
    
    def submit_frame(self, frame):
        """
        Submit a frame for processing (non-blocking)
        Returns immediately, result available via get_result()
        """
        self._frame_count += 1
        
        # Skip frames for performance
        if self._frame_count % self.skip_frames != 0:
            return
        
        # Replace any pending frame with the new one (drop old frames)
        try:
            # Clear the queue if not empty
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Add new frame
            self._input_queue.put_nowait((frame.copy(), self._frame_count))
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def _process_loop(self):
        """Background processing loop"""
        while self._running:
            try:
                # Wait for a frame with timeout
                frame, frame_id = self._input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Process the frame
                fpn_features = self.vision.process_frame(frame)
                
                if fpn_features is not None:
                    # Run policy network
                    action, confidence, probs = self.policy.predict(fpn_features)
                    
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
                
                # Update performance metrics
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
                self.logger.error(f"Vision processing error: {e}")
    
    def get_result(self, max_age=0.5):
        """
        Get the latest vision result (thread-safe)
        
        Args:
            max_age: Maximum age of result in seconds (returns None if older)
            
        Returns:
            dict with action, confidence, probs or None if no recent result
        """
        with self._lock:
            if self._latest_result is None:
                return None
            
            age = time.time() - self._result_timestamp
            if age > max_age:
                return None
            
            return self._latest_result.copy()
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            'fps': self._fps,
            'processing_time_ms': self._processing_time * 1000,
            'frames_submitted': self._frame_count,
            'frames_processed': self._processed_count,
            'skip_frames': self.skip_frames
        }
    
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
    """
    Continuous camera stream for real-time processing
    Optimized with threaded capture to prevent frame blocking
    """
    
    def __init__(self, device='/dev/video0', width=640, height=480, buffer_size=2):
        self.device = device
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Threading for non-blocking capture
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None
    
    def start(self):
        """Start camera stream with threaded capture"""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open camera: {self.device}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Try to set higher FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        self.logger.info(f"Camera stream started: {self.device}")
        return True
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
    
    def read(self):
        """Read the latest frame (non-blocking)"""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def read_with_id(self):
        """Read frame with its ID to detect new frames"""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_id
            return None, 0
    
    def stop(self):
        """Stop camera stream"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Camera stream stopped")

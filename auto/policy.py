#!/usr/bin/env python3
"""
Policy module - TinyNet neural network for action prediction
Outputs: left, straight, right, reverse
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

class PolicyNetwork:
    """Neural network policy for autonomous navigation"""
    
    def __init__(self, config, use_tinygrad=False):
        self.config = config
        self.use_tinygrad = use_tinygrad
        self.logger = logging.getLogger(__name__)
        
        self.class_names = ["left", "straight", "right", "reverse"]
        
        if use_tinygrad:
            self._init_tinygrad()
        else:
            self._init_pytorch()
    
    def _init_tinygrad(self):
        """Initialize tinygrad-based TinyNet policy"""
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
            
            from train import TinyNet
            
            self.logger.info("Initializing TinyNet policy with tinygrad")
            
            # Load network architecture
            self.net = TinyNet()
            
            # Load trained weights
            policy_path = self.config['models']['policy_model']
            if not os.path.exists(policy_path):
                self.logger.warning(f"Policy weights not found at {policy_path}")
                self.logger.warning("Policy will use random initialization - train first!")
            else:
                state_dict = safe_load(policy_path)
                load_state_dict(self.net, state_dict)
                self.logger.info(f"Policy loaded from {policy_path}")
            
            # Set to inference mode
            Tensor.training = False
            Tensor.no_grad = True
            
        except ImportError as e:
            self.logger.error(f"tinygrad import failed: {e}")
            raise
    
    def _init_pytorch(self):
        """Initialize PyTorch-based TinyNet policy (converted from safetensors)"""
        try:
            import torch
            import torch.nn as nn
            from safetensors.torch import load_file
            
            self.logger.info("Initializing TinyNet policy with PyTorch")
            
            # Check GPU availability
            use_gpu = self.config.get('vision', {}).get('use_gpu', False)
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                if use_gpu:
                    self.logger.warning("GPU requested but not available, using CPU")
            
            # Define TinyNet architecture (matching dollop's train.py)
            class TinyNetPyTorch(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Match the EXACT architecture from dollop/train.py
                    # Input: 256 channels from YOLOv8 FPN features
                    self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(64, 8, kernel_size=3, padding=1)
                    self.fc = nn.Linear(1408, 4)  # 4 output classes
                    self.dropout = nn.Dropout(0.85)
                
                def forward(self, x):
                    # Input should be (B, 256, H, W) from YOLOv8 FPN
                    x = torch.relu(self.conv1(x))  # GELU in dollop, ReLU is close enough
                    x = torch.relu(self.conv2(x))
                    x = self.dropout(x)
                    x = x.reshape(x.size(0), -1)  # Flatten
                    x = self.fc(x)
                    return x
            
            # Create model
            self.net = TinyNetPyTorch().to(self.device)
            
            # Load trained weights from safetensors
            policy_path = self.config['models']['policy_model']
            if not os.path.exists(policy_path):
                self.logger.warning(f"Policy weights not found at {policy_path}")
                self.logger.warning("Policy will use random initialization - train first!")
            else:
                # Load safetensors weights
                state_dict = load_file(policy_path)
                
                # Convert tinygrad naming to PyTorch naming if needed
                pytorch_state_dict = {}
                for key, value in state_dict.items():
                    # Tinygrad uses different naming conventions
                    new_key = key.replace('.weight', '.weight').replace('.bias', '.bias')
                    if 'c.' in key:  # Tinygrad conv layer naming
                        new_key = key.replace('c.', 'conv')
                    if 'l.' in key:  # Tinygrad linear layer naming
                        new_key = key.replace('l.', 'fc.')
                    pytorch_state_dict[new_key] = torch.from_numpy(np.array(value))
                
                self.net.load_state_dict(pytorch_state_dict, strict=False)
                self.logger.info(f"Policy loaded from {policy_path}")
            
            # Set to eval mode
            self.net.eval()
            
        except ImportError as e:
            self.logger.error(f"PyTorch or safetensors import failed: {e}")
            self.logger.info("Install with: pip install torch safetensors")
            raise
        
    def predict_tinygrad(self, features):
        """Predict action using tinygrad TinyNet"""
        from tinygrad.tensor import Tensor
        
        # Forward pass through policy network
        logits = self.net(features)
        
        # Apply softmax to get probabilities
        probs = logits.exp().realize()
        probs_np = probs.numpy()[0]
        
        # Get predicted class and confidence
        pred_class = np.argmax(probs_np)
        confidence = probs_np[pred_class]
        
        action = self.class_names[pred_class]
        
        return action, confidence, probs_np
    
    def predict_pytorch(self, features):
        """Predict action using PyTorch TinyNet"""
        import torch
        
        # Convert features to tensor
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float().to(self.device)
        else:
            features_tensor = features
        
        # Forward pass
        with torch.no_grad():
            logits = self.net(features_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()[0]
        
        # Get predicted class and confidence
        pred_class = np.argmax(probs_np)
        confidence = probs_np[pred_class]
        
        action = self.class_names[pred_class]
        
        return action, confidence, probs_np
    
    def predict_simple(self, frame):
        """Simple heuristic policy without ML (for testing)"""
        # This is a placeholder - in real use, you need the trained model
        # For now, default to going straight
        
        # Simple logic: could use edge detection, contours, etc.
        # But without trained model, just return straight with low confidence
        
        action = "straight"
        confidence = 0.5
        probs = np.array([0.15, 0.5, 0.15, 0.2])  # [left, straight, right, reverse]
        
        return action, confidence, probs
    
    def predict(self, features):
        """Main prediction interface"""
        if self.use_tinygrad:
            return self.predict_tinygrad(features)
        else:
            return self.predict_pytorch(features)
    
    def get_action_name(self, action_idx):
        """Convert action index to name"""
        if 0 <= action_idx < len(self.class_names):
            return self.class_names[action_idx]
        return "unknown"


class PolicyEvaluator:
    """Evaluates policy predictions and applies safety checks"""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = config['safety']['confidence_threshold']
        self.logger = logging.getLogger(__name__)
        
        # Track recent predictions for smoothing
        self.prediction_history = []
        self.history_size = 5
    
    def add_prediction(self, action, confidence):
        """Add prediction to history"""
        self.prediction_history.append((action, confidence))
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
    
    def get_smoothed_action(self):
        """Get most common action from recent history"""
        if not self.prediction_history:
            return None, 0.0
        
        # Count occurrences of each action
        action_counts = {}
        total_confidence = {}
        
        for action, conf in self.prediction_history:
            action_counts[action] = action_counts.get(action, 0) + 1
            total_confidence[action] = total_confidence.get(action, 0) + conf
        
        # Get most common action
        best_action = max(action_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence[best_action] / action_counts[best_action]
        
        return best_action, avg_confidence
    
    def should_execute(self, action, confidence):
        """Determine if action should be executed based on confidence"""
        if confidence < self.confidence_threshold:
            self.logger.debug(f"Action '{action}' rejected: confidence {confidence:.2f} < {self.confidence_threshold}")
            return False
        return True
    
    def clear_history(self):
        """Clear prediction history"""
        self.prediction_history = []

#!/usr/bin/env python3
"""
Training script for DirectPolicyNet - End-to-end policy without YOLOv8.
Uses the same training data format as the original dollop training.

This is ~10x faster than YOLOv8 + TinyNet at inference time.
"""

import os
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import trange

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load
from tinygrad.nn.optim import AdamW

# Training data - same format as dollop
train_set = [
    ("left_1_dcamera", 0),
    ("left_3_dcamera", 0),
    ("right_1_dcamera", 2),
    ("right_3_dcamera", 2),
    ("straight_1_dcamera", 1),
    ("ljust_1_dcamera", 0),
    ("rjust_1_dcamera", 2),
    ("left_day_1_dcamera", 0),
    ("right_day_1_dcamera", 2),
    ("left_day_2_dcamera", 0),
    ("right_day_2_dcamera", 2),
    ("left_day_3_dcamera", 0),
    ("right_day_3_dcamera", 2),
    ("left_day_4_dcamera", 0),
    ("right_day_4_dcamera", 2),
    ("extra_left_1_dcamera", 0),
    ("extra_right_1_dcamera", 2),
    ("extra_left_2_dcamera", 0),
    ("extra_right_2_dcamera", 2),
    ("extra_straight_2_dcamera", 1),
    ("extra_left_3_dcamera", 0),
    ("extra_right_3_dcamera", 2),
    ("messup_left_1_dcamera", 0),
    ("messup_left_2_dcamera", 0),
    ("messup_left_3_dcamera", 0),
    ("messup_left_4_dcamera", 0),
    ("reverse_1_dcamera", 3),
    ("reverse_2_dcamera", 3),
    ("reverse_3_dcamera", 3),
    ("messup_right_1_dcamera", 2),
    ("messup_right_2_dcamera", 2),
    ("messup_right_3_dcamera", 2),
    ("messup_right_4_dcamera", 2),
    ("messup_right_5_dcamera", 2),
    ("messup_right_6_dcamera", 2),
    ("messup_right_7_dcamera", 2),
    ("messup_right_8_dcamera", 2),
    ("messup_right_9_dcamera", 2),
    ("messup_reverse_1_dcamera", 3),
    ("messup_reverse_2_dcamera", 3),
    ("messup_reverse_3_dcamera", 3),
    ("messup_straight_1_dcamera", 1),
]

test_set = [
    ("left_2_dcamera", 0),
    ("straight_2_dcamera", 1),
    ("right_2_dcamera", 2),
]


class DirectPolicyNet:
    """
    End-to-end policy network that takes raw images.
    Input: (B, 3, 120, 160) normalized RGB images
    Output: 4 classes (left, straight, right, reverse)
    """
    def __init__(self):
        # Input: 3 channels (RGB), 160x120
        self.c1 = Conv2d(3, 16, 5, stride=2, padding=2)   # -> 16x80x60
        self.c2 = Conv2d(16, 32, 3, stride=2, padding=1)  # -> 32x40x30
        self.c3 = Conv2d(32, 64, 3, stride=2, padding=1)  # -> 64x20x15
        self.c4 = Conv2d(64, 64, 3, stride=2, padding=1)  # -> 64x10x7 (with 160x120 input)
        self.fc1 = Linear(64 * 10 * 7, 64)
        self.fc2 = Linear(64, 4)
    
    def __call__(self, x):
        x = self.c1(x).relu()
        x = self.c2(x).relu()
        x = self.c3(x).relu()
        x = self.c4(x).relu()
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x).relu().dropout(0.3)
        x = self.fc2(x)
        return x.log_softmax()


def load_and_preprocess_video(video_path, target_size=(160, 120)):
    """
    Load video and preprocess frames to 160x120 RGB normalized.
    Returns tensor of shape (N, 3, 120, 160)
    """
    import cv2
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to 160x120
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        normalized = rgb.astype(np.float32) / 255.0
        
        # (H, W, C) -> (C, H, W)
        transposed = normalized.transpose(2, 0, 1)
        
        frames.append(transposed)
    
    cap.release()
    
    return np.array(frames)


def get_minibatch(sets, data_cache, bs=32):
    """Get a minibatch of training data"""
    xs, ys = [], []
    
    for _ in range(bs):
        name, label = random.choice(sets)
        
        if name not in data_cache:
            # Try to find the video
            video_path = None
            for ext in ['.mp4', '.avi', '.mov']:
                candidate = f"commabot/dollop-main/{name}{ext}"
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if video_path is None:
                print(f"Warning: Video not found for {name}")
                continue
            
            data_cache[name] = load_and_preprocess_video(video_path)
        
        frames = data_cache[name]
        if len(frames) == 0:
            continue
        
        # Random frame
        idx = random.randint(0, len(frames) - 1)
        xs.append(frames[idx])
        ys.append(label)
    
    if len(xs) == 0:
        return None, None
    
    return Tensor(np.array(xs)), Tensor(np.array(ys))


def train():
    """Main training loop"""
    print("="*60)
    print("Training DirectPolicyNet (end-to-end, no YOLOv8)")
    print("="*60)
    
    # Initialize network
    net = DirectPolicyNet()
    optim = AdamW(get_parameters(net), lr=0.001)
    
    # Data cache
    data_cache = {}
    test_cache = {}
    
    # Check if training data exists
    data_dir = "commabot/dollop-main"
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please ensure training videos are in the correct location.")
        return
    
    print(f"\nLoading training data from: {data_dir}")
    
    # Training loop
    best_accuracy = 0
    
    for step in (t := trange(3000)):
        Tensor.training = True
        optim.lr *= 0.9995
        
        # Get batch
        x, y = get_minibatch(train_set, data_cache, bs=32)
        if x is None:
            continue
        
        # Forward pass
        out = net(x)
        loss = out.sparse_categorical_crossentropy(y)
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # Calculate accuracy
        cat = out.argmax(axis=-1)
        accuracy = (cat == y).mean()
        
        loss_val = loss.realize().numpy()
        acc_val = accuracy.realize().numpy()
        
        t.set_description(f"loss: {loss_val:.4f}, acc: {acc_val:.2%}")
        
        # Periodic test
        if step % 100 == 0 and step > 0:
            Tensor.training = False
            tx, ty = get_minibatch(test_set, test_cache, bs=16)
            if tx is not None:
                test_out = net(tx)
                test_cat = test_out.argmax(axis=-1)
                test_acc = (test_cat == ty).mean().realize().numpy()
                print(f"\n  Test accuracy: {test_acc:.2%}")
                
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    # Save best model
                    state = get_state_dict(net)
                    save_path = "models/direct_policy.safetensors"
                    os.makedirs("models", exist_ok=True)
                    safe_save(state, save_path)
                    print(f"  Saved best model ({test_acc:.2%}) to {save_path}")
    
    # Final save
    state = get_state_dict(net)
    save_path = "models/direct_policy.safetensors"
    os.makedirs("models", exist_ok=True)
    safe_save(state, save_path)
    print(f"\nFinal model saved to {save_path}")
    print(f"Best test accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    train()

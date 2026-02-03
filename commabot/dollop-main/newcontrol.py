#!/usr/bin/env python3
import time
import numpy as np
import cv2
from pathlib import Path
import sys

# tinygrad imports
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.helpers import Timing      
from tinygrad.dtype import dtypes       #dtypes is in another dtypes file and not in helpers anymore

# repo local imports
from train import TinyNet
from yolov8 import YOLOv8, get_variant_multiples
import requests

def download_file(url, out_path):
    print(f"Downloading {url} to {out_path}")
    r = requests.get(url)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        f.write(r.content)

def get_foundation():
    yolo_variant = "n"
    depth, width, ratio = get_variant_multiples(yolo_variant)
    yolo_infer = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)

    weights_location = Path("yolov8n.safetensors")
    if not weights_location.exists():
        print(
            f"\nERROR: YOLOv8 weights file not found at {weights_location}\n"
            "Please manually download yolov8n.safetensors from:\n"
            "  https://gitlab.com/r3sist/yolov8_weights/-/raw/master/yolov8n.safetensors\n"
            f"And place it in: {weights_location}\n"
        )
        sys.exit(1)

    state_dict = safe_load(weights_location)
    load_state_dict(yolo_infer, state_dict)

    def foundation(imgs: Tensor):
        x = yolo_infer.net(imgs.permute(0,3,1,2).float() / 255)
        x = yolo_infer.fpn(*x)
        return x[2]
    return foundation

def get_tinynet():
    weights_path = Path("tinynet.safetensors")
    if not weights_path.exists():
        print(
            f"\nERROR: TinyNet weights file not found at {weights_path}\n"
            "You need to train the model using train.py, or place the weights file here.\n"
        )
        sys.exit(1)
    net = TinyNet()
    load_state_dict(net, safe_load(weights_path))
    return net

def get_pred():
    Tensor.no_grad = True
    Tensor.training = False
    foundation = get_foundation()
    net = get_tinynet()
    def pred(x):
        return net(foundation(x)).exp()[0].realize()
    # Warm up
    with Timing("building 1: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.float32))
    with Timing("building 2: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.float32))
    with Timing("testing rt: "): pred(Tensor.rand(1,480,640,3, dtype=dtypes.float32)).numpy()
    return pred

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pred = get_pred()
    action_names = ["left", "straight", "right", "reverse"]

    print("Press 'q' to quit.")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Webcam error.")
            break
        img = cv2.resize(img, (640, 480))

        probs = pred(Tensor(img).reshape(1, 480, 640, 3)).numpy()
        action = np.argmax(probs)
        print(f"Probabilities: {probs}, Predicted action: {action_names[action]}")

        cv2.putText(img, f"Action: {action_names[action]}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Webcam Demo", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
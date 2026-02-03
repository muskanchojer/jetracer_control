#works but see flip
"""
Webcam-only live predictions, matched to geohot/body_loop video path:
- Convert BGR -> RGB
- DO NOT divide by 255 here (video.py yields uint8 RGB via rgb24)
- Resize to 640x480 (INTER_AREA)
- Feed NHWC (1, H, W, 3) to tinygrad model
- Optional YUV-limited simulation to mimic device/decoder (BT.601/709)

Usage:
  python live_webcam_repo_match.py
  python live_webcam_repo_match.py --cam 1
  python live_webcam_repo_match.py --flip
  python live_webcam_repo_match.py --simulate-yuv 1 --matrix 709
"""

import argparse
import time
import numpy as np
import cv2
from tinygrad.tensor import Tensor
from newcontrol import get_pred

CLASS_NAMES = ["left", "straight", "right", "reverse"]
FRAME_W, FRAME_H = 640, 480

# ----------------- Optional YUV(limited) helpers (remain uint8 end-to-end) -----------------

def rgb8_to_yuv8_limited(rgb8: np.ndarray, matrix: str = "601") -> np.ndarray:
    rgb = rgb8.astype(np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if matrix == "709":
        Y = 0.183 * R + 0.614 * G + 0.062 * B + 16.0
        U = -0.101 * R - 0.339 * G + 0.439 * B + 128.0
        V = 0.439 * R - 0.399 * G - 0.040 * B + 128.0
    else:
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
        U = -0.148 * R - 0.291 * G + 0.439 * B + 128.0
        V = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
    yuv = np.stack([Y, U, V], axis=-1)
    np.clip(yuv, 0, 255, out=yuv)
    return yuv.astype(np.uint8)

def yuv8_limited_to_rgb8(yuv8: np.ndarray, matrix: str = "601") -> np.ndarray:
    yuv = yuv8.astype(np.float32)
    Y = yuv[..., 0] - 16.0
    U = yuv[..., 1] - 128.0
    V = yuv[..., 2] - 128.0
    C = 1.164  # 255/219
    if matrix == "709":
        R = C * Y + 1.793 * V
        G = C * Y - 0.213 * U - 0.534 * V
        B = C * Y + 2.115 * U
    else:
        R = C * Y + 1.596 * V
        G = C * Y - 0.392 * U - 0.813 * V
        B = C * Y + 2.017 * U
    rgb = np.stack([R, G, B], axis=-1)
    np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.uint8)

def simulate_yuv_roundtrip_rgb(rgb8: np.ndarray, matrix: str = "601") -> np.ndarray:
    yuv8 = rgb8_to_yuv8_limited(rgb8, matrix=matrix)
    return yuv8_limited_to_rgb8(yuv8, matrix=matrix)

# ----------------- Preprocessing: mirror repo’s RGB24 (no /255 here) -----------------

def preprocess_webcam_bgr_to_rgb8(
    bgr_frame: np.ndarray,
    simulate_yuv: bool = False,
    yuv_matrix: str = "601",
    flip_horizontal: bool = True,
) -> np.ndarray:
    """
    BGR uint8 -> (optional flip) -> RGB uint8 -> (optional YUV-limited RT) ->
    resize to 640x480 (INTER_AREA) -> uint8 RGB (H, W, 3)
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Optional flip if orientation differs
    if flip_horizontal:
        rgb = cv2.flip(rgb, 1)

    # Optional emulate device/decoder YUV-limited pipeline
    if simulate_yuv:
        rgb = simulate_yuv_roundtrip_rgb(rgb, matrix=yuv_matrix)

    # Resize to exact input
    if rgb.shape[1] != FRAME_W or rgb.shape[0] != FRAME_H:
        rgb = cv2.resize(rgb, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    # Keep as uint8 [0,255] to match video.py (rgb24) path
    return np.ascontiguousarray(rgb, dtype=np.uint8)

def overlay_prediction(rgb8: np.ndarray, label: str, probs: np.ndarray) -> np.ndarray:
    """Draw overlay on BGR for imshow."""
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    prob_text = " | ".join([f"{n}: {p:.2f}" for n, p in zip(CLASS_NAMES, probs)])
    cv2.putText(bgr, f"Pred: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(bgr, prob_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--flip", action="store_true", help="Flip horizontally before inference")
    ap.add_argument("--simulate-yuv", type=int, default=0, help="Simulate YUV-limited round-trip (0/1)")
    ap.add_argument("--matrix", choices=["601", "709"], default="601", help="YUV matrix for simulation")
    ap.add_argument("--no-show", action="store_true", help="Run headless")
    ap.add_argument("--debug", action="store_true", help="Print first-frame dtype/range stats")
    args = ap.parse_args()

    # Open webcam; CAP_DSHOW tends to be more stable on Windows
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(FRAME_W))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(FRAME_H))

    if not cap.isOpened():
        print("Cannot open camera. Check index/permissions.")
        return

    pred = get_pred()
    printed_stats = False
    t0, frames = time.time(), 0
    print("Live prediction running. Press 'q' to quit." if not args.no_show else "Live prediction (headless).")

    while True:
        ok, bgr = cap.read()
        if not ok:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        rgb8 = preprocess_webcam_bgr_to_rgb8(
            bgr,
            simulate_yuv=bool(args.simulate_yuv),
            yuv_matrix=args.matrix,
            flip_horizontal=args.flip,
        )  # uint8 RGB [0..255], shape (480, 640, 3)

        if args.debug and not printed_stats:
            ch_mean = rgb8.mean(axis=(0,1))
            ch_min = rgb8.min(axis=(0,1))
            ch_max = rgb8.max(axis=(0,1))
            print(f"[DEBUG] rgb8 dtype={rgb8.dtype}, shape={rgb8.shape}, "
                  f"mean={ch_mean}, min={ch_min}, max={ch_max}")
            printed_stats = True

        # NHWC -> batch. IMPORTANT: keep as uint8 (NO /255), like video.py (rgb24)
        input_tensor = Tensor(rgb8.reshape(1, FRAME_H, FRAME_W, 3))

        # Inference
        probs = pred(input_tensor).numpy().copy().flatten()
        label = CLASS_NAMES[int(np.argmax(probs))]

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            print(f"FPS ~ {fps:.2f} | Pred: {label} | Probs: {np.round(probs, 3)}")
            t0, frames = now, 0

        if not args.no_show:
            disp = overlay_prediction(rgb8, label, probs)
            cv2.imshow("Live Camera Prediction (Repo-Matched Pipeline)", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
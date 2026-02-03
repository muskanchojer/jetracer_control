#!/usr/bin/env python3
"""
Live webcam predictions (repo-aligned) with:
- Flip auto-calibration wizard (--auto-calibrate or press 'c')
- Capture thread + latest-frame queue to avoid blocking
- Inference stride (--stride) to increase UI FPS (reuse last prediction between inferences)
- RGB uint8 pipeline (no external /255), 640x480, NHWC

Keys:
  q : quit
  f : toggle flip
  c : run auto-calibration wizard

Usage examples:
  python live_webcam_calibrated.py --auto-calibrate
  python live_webcam_calibrated.py --stride 2
  python live_webcam_calibrated.py --flip 1 --stride 3
"""

import argparse
import time
import threading
import queue
import numpy as np
import cv2

from tinygrad.tensor import Tensor
from newcontrol import get_pred

CLASS_NAMES = ["left", "straight", "right", "reverse"]
FRAME_W, FRAME_H = 640, 480

# ----------------- Preprocessing (repo-aligned RGB uint8) -----------------

def preprocess_webcam_bgr_to_rgb8(bgr_frame: np.ndarray, flip_horizontal: bool) -> np.ndarray:
    """BGR uint8 -> (optional flip) -> RGB uint8 -> resize (INTER_AREA) -> RGB uint8."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if flip_horizontal:
        rgb = cv2.flip(rgb, 1)
    if rgb.shape[1] != FRAME_W or rgb.shape[0] != FRAME_H:
        rgb = cv2.resize(rgb, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb, dtype=np.uint8)

def to_tensor_preserve_scale(arr: np.ndarray) -> Tensor:
    """Create tinygrad Tensor keeping numeric scale (uint8 0..255)."""
    H, W, C = arr.shape
    return Tensor(arr.reshape(1, H, W, C))

# ----------------- Overlay & HUD -----------------

def overlay_prediction(rgb8: np.ndarray, label: str, probs: np.ndarray, flip_on: bool, fps: float, mode: str=None, prompt: str=None) -> np.ndarray:
    """Draw overlay text on BGR for imshow."""
    bgr = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    prob_text = " | ".join([f"{n}: {p:.2f}" for n, p in zip(CLASS_NAMES, probs)])
    y = 24
    if mode:
        cv2.putText(bgr, f"{mode}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2); y += 28
    cv2.putText(bgr, f"Pred: {label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2); y += 32
    cv2.putText(bgr, prob_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2); y += 28
    cv2.putText(bgr, f"Flip: {'ON' if flip_on else 'OFF'} | FPS: {fps:.2f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2); y += 28
    if prompt:
        for i, line in enumerate(prompt.split("\n")):
            cv2.putText(bgr, line, (10, y+28*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
    return bgr

# ----------------- Capture Thread -----------------

def start_capture_thread(cam_index: int, width: int, height: int, buffer_size: int = 1):
    """
    Returns: (cap, q, stop_fn)
      - cap: cv2.VideoCapture
      - q  : queue.Queue with the latest BGR frame (maxsize=buffer_size)
      - stop_fn: function to stop the thread
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    # Try to reduce internal buffering (may be ignored by some drivers)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    qf = queue.Queue(maxsize=buffer_size)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            # Put latest frame, drop older if queue full
            if qf.full():
                try: qf.get_nowait()
                except Exception: pass
            qf.put(bgr, block=False)

    th = threading.Thread(target=run, daemon=True)
    th.start()

    def stop_fn():
        stop.set()
        th.join(timeout=1.0)
        cap.release()

    return cap, qf, stop_fn

# ----------------- Flip Auto-Calibration Wizard -----------------

def predict_two_orientations(pred, rgb8_no_flip: np.ndarray):
    """Return (label_no_flip, probs_no_flip), (label_flip, probs_flip)."""
    # No flip path: use as-is
    t_noflip = to_tensor_preserve_scale(rgb8_no_flip)
    pA = pred(t_noflip).numpy().copy().flatten()
    lblA = CLASS_NAMES[int(np.argmax(pA))]

    # Flip path: flip the RGB and run again
    rgb8_flip = cv2.flip(rgb8_no_flip, 1)
    t_flip = to_tensor_preserve_scale(rgb8_flip)
    pB = pred(t_flip).numpy().copy().flatten()
    lblB = CLASS_NAMES[int(np.argmax(pB))]

    return (lblA, pA), (lblB, pB)

def calibrate_flip(pred, frame_src_queue: queue.Queue, show_window: bool = True) -> bool:
    """
    Guided, two-stage calibration. Returns best flip flag (True/False).
    Stage 1 (CENTER): Point camera centered; press SPACE to sample N frames.
      - Choose orientation with higher avg P(straight).
    Stage 2 (TOO FAR LEFT): Move left (as in training "left" meaning -> should go RIGHT); press SPACE to sample N frames.
      - Choose orientation with more 'right' predictions.
    Final: Majority/score combine. Esc cancels (keeps current flip).
    """
    N = 24
    best_flip = False
    win = "Live Camera Prediction (calibrating)"
    tmp_fps = 0.0
    stage = 0
    score = {"noflip_straight":0.0, "flip_straight":0.0, "noflip_right":0, "flip_right":0}

    instructions = [
        "CALIBRATION STAGE 1/2: CENTER VIEW\n"
        "Point camera straight/centered.\n"
        "Press SPACE to capture ~24 samples.",

        "CALIBRATION STAGE 2/2: TOO FAR LEFT VIEW\n"
        "Move camera/robot slightly to the LEFT side (as if too far left); expected action is RIGHT.\n"
        "Press SPACE to capture ~24 samples.",
    ]

    last_rgb = None
    while True:
        try:
            bgr = frame_src_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        rgb8 = preprocess_webcam_bgr_to_rgb8(bgr, flip_horizontal=False)
        last_rgb = rgb8.copy()

        # Show instruction frame
        if show_window:
            disp = overlay_prediction(rgb8, label="(calibrating)", probs=np.array([0,0,0,0], dtype=np.float32),
                                      flip_on=False, fps=tmp_fps, mode="CALIBRATION", prompt=instructions[stage])
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255  # no-op

        if key == 27:  # ESC cancels
            return best_flip
        if key == ord('q'):
            return best_flip
        if key == 32:  # SPACE: sample
            # Sample N frames for this stage
            straight_noflip = []
            straight_flip = []
            right_noflip = 0
            right_flip = 0

            for _ in range(N):
                try:
                    bgr2 = frame_src_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                rgb8_2 = preprocess_webcam_bgr_to_rgb8(bgr2, flip_horizontal=False)

                (lblA, pA), (lblB, pB) = predict_two_orientations(pred, rgb8_2)

                # Record stats
                straight_noflip.append(pA[CLASS_NAMES.index("straight")])
                straight_flip.append(pB[CLASS_NAMES.index("straight")])
                if lblA == "right": right_noflip += 1
                if lblB == "right": right_flip += 1

                # Small visual feedback
                if show_window:
                    disp = overlay_prediction(rgb8_2, label=f"A:{lblA} | B(flip):{lblB}",
                                              probs=pA, flip_on=False, fps=tmp_fps,
                                              mode=f"CALIBRATING ({_+1}/{N})", prompt=None)
                    cv2.imshow(win, disp)
                    cv2.waitKey(1)

            # Aggregate for this stage
            if stage == 0:
                score["noflip_straight"] = float(np.mean(straight_noflip)) if len(straight_noflip) else 0.0
                score["flip_straight"]   = float(np.mean(straight_flip))   if len(straight_flip) else 0.0
            else:
                score["noflip_right"] = right_noflip
                score["flip_right"]   = right_flip

            # Next stage or finish
            stage += 1
            if stage >= 2:
                break

    # Decision logic:
    #  - Primary: Stage 2 expectation — when "too far left", the correct action is "right".
    #  - Secondary: Stage 1 — choose orientation with higher P(straight) in centered view.
    #  - Tiebreaker: sum normalized scores.
    prefer_flip_by_stage2 = score["flip_right"] > score["noflip_right"]
    prefer_flip_by_stage1 = score["flip_straight"] > score["noflip_straight"]

    if prefer_flip_by_stage2 and prefer_flip_by_stage1:
        best_flip = True
    elif (not prefer_flip_by_stage2) and (not prefer_flip_by_stage1):
        best_flip = False
    else:
        # Mixed signals: combine scores
        # Normalize straight score into [0,1] by dividing by max observed
        denom = max(score["flip_straight"], score["noflip_straight"], 1e-6)
        flip_straight_norm   = score["flip_straight"]   / denom
        noflip_straight_norm = score["noflip_straight"] / denom

        # Normalize right counts by N
        N = max(score["flip_right"] + score["noflip_right"], 1)
        flip_right_norm   = score["flip_right"]   / N
        noflip_right_norm = score["noflip_right"] / N

        flip_total   = 0.6 * flip_right_norm   + 0.4 * flip_straight_norm
        noflip_total = 0.6 * noflip_right_norm + 0.4 * noflip_straight_norm
        best_flip = flip_total > noflip_total

    # One last feedback frame
    if show_window and last_rgb is not None:
        msg = (f"CALIBRATION COMPLETE\n"
               f"noflip: straight={score['noflip_straight']:.3f}, right_cnt={score['noflip_right']}\n"
               f"flip  : straight={score['flip_straight']:.3f}, right_cnt={score['flip_right']}\n"
               f"=> Using Flip: {'ON' if best_flip else 'OFF'}")
        disp = overlay_prediction(last_rgb, label="(calibrated)", probs=np.array([0,0,0,0], dtype=np.float32),
                                  flip_on=best_flip, fps=0.0, mode="CALIBRATION", prompt=msg)
        cv2.imshow(win, disp); cv2.waitKey(800)

    return bool(best_flip)

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--flip", type=int, default=1, help="Start with flip ON(1)/OFF(0). Default 1.")
    ap.add_argument("--auto-calibrate", action="store_true", help="Run flip auto-calibration at startup")
    ap.add_argument("--stride", type=int, default=1, help="Run inference every N frames; reuse last prediction in-between. Default 1 (no skipping).")
    ap.add_argument("--no-show", action="store_true", help="Run headless (no window)")
    ap.add_argument("--opencv-threads", type=int, default=1, help="OpenCV threads (often 1 is best for small ops)")
    args = ap.parse_args()

    # Reduce OpenCV threading overhead (often helps a bit)
    try: cv2.setNumThreads(int(args.opencv_threads))
    except Exception: pass

    # Capture thread (avoid camera read stall in main thread)
    cap, frame_q, stop_cap = start_capture_thread(args.cam, FRAME_W, FRAME_H)

    pred = get_pred()
    flip_on = bool(args.flip)

    # Create window up-front
    if not args.no_show:
        cv2.namedWindow("Live Camera Prediction", cv2.WINDOW_NORMAL)

    # First-frame debug + optional auto-calibration
    first_rgb = None
    while first_rgb is None:
        try:
            bgr0 = frame_q.get(timeout=1.0)
        except queue.Empty:
            print("Waiting for first frame...")
            continue
        first_rgb = preprocess_webcam_bgr_to_rgb8(bgr0, flip_horizontal=False)

    # Auto-calibrate if requested
    if args.auto_calibrate and not args.no_show:
        try:
            flip_on = calibrate_flip(pred, frame_q, show_window=True)
            print(f"[CALIB] Flip chosen -> {'ON' if flip_on else 'OFF'}")
        except Exception as e:
            print(f"[CALIB] Calibration skipped due to error: {e}")

    # Main loop
    print("Running. Keys: q=quit, f=toggle flip, c=calibrate flip")
    t0 = time.time(); frames_disp = 0
    fps_disp = 0.0
    frame_idx = 0
    last_label = "..."
    last_probs = np.array([0,0,0,0], dtype=np.float32)
    last_rgb_shown = first_rgb

    try:
        while True:
            try:
                bgr = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue

            rgb8 = preprocess_webcam_bgr_to_rgb8(bgr, flip_horizontal=flip_on)

            # Inference stride (compute every Nth frame)
            compute_now = (frame_idx % max(1, args.stride) == 0)
            if compute_now:
                t = to_tensor_preserve_scale(rgb8)
                probs = pred(t).numpy().copy().flatten()
                label = CLASS_NAMES[int(np.argmax(probs))]
                last_probs = probs
                last_label = label

            # FPS estimate (UI)
            frames_disp += 1
            now = time.time()
            if now - t0 >= 0.5:
                fps_disp = frames_disp / (now - t0)
                t0 = now; frames_disp = 0

            # Display
            if not args.no_show:
                disp = overlay_prediction(rgb8, last_label, last_probs, flip_on=flip_on, fps=fps_disp)
                cv2.imshow("Live Camera Prediction", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    flip_on = not flip_on
                    print(f"[INFO] Flip toggled -> {'ON' if flip_on else 'OFF'}")
                elif key == ord('c'):
                    try:
                        flip_on = calibrate_flip(pred, frame_q, show_window=True)
                        print(f"[CALIB] Flip chosen -> {'ON' if flip_on else 'OFF'}")
                    except Exception as e:
                        print(f"[CALIB] Calibration error: {e}")
            else:
                # Headless print (throttle a bit)
                if compute_now:
                    print(f"Flip={'ON' if flip_on else 'OFF'} | Pred: {last_label} | Probs: {np.round(last_probs, 3)} | FPS~{fps_disp:.2f}")

            frame_idx += 1

    finally:
        stop_cap()
        if not args.no_show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
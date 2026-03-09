#!/usr/bin/env python3
"""
Simple OpenCV camera viewer — runs on the HOST (not inside Docker).
Opens /dev/video0 and shows a live window. Press Q or ESC to close.
"""

import cv2
import sys

DEVICE = 0          # /dev/video0
WIDTH  = 640
HEIGHT = 480
FPS    = 15
WINDOW = "JetRacer Camera"


def main():
    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        print(f"❌ Cannot open camera (device {DEVICE})")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    print(f"📷 Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{int(cap.get(cv2.CAP_PROP_FPS))} fps")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

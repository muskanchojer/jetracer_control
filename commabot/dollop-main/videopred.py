#!/usr/bin/env python3
#gives correct model outputs on video files

import sys
import cv2
import numpy as np
from newcontrol import get_pred
from video import frames_from_file
from tinygrad.tensor import Tensor

def main():
    # Use sys.argv to get the video file path, default to a sample file
    video_path = "data/botcam.avi" if len(sys.argv) == 1 else sys.argv[1]

    pred = get_pred()
    
    FRAME_HEIGHT = 480
    FRAME_WIDTH = 640
    class_names = ["left", "straight", "right", "reverse"]

    print(f"Processing video file: {video_path}")
    print("Press 'q' to quit the display.")
    
    for i, frm in enumerate(frames_from_file(video_path)):
        # The frames_from_file function already provides preprocessed frames
        # in the correct shape (480, 640, 3) and type (float32, 0-1 range).
        # We just need to reshape for the model's batch dimension.
        input_tensor = Tensor(frm.reshape(1, FRAME_HEIGHT, FRAME_WIDTH, 3))

        # Get prediction probabilities
        probs = pred(input_tensor).numpy().copy().flatten()

        # Display probabilities and predicted class
        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]
        prob_text = " | ".join([f"{name}: {prob:.2f}" for name, prob in zip(class_names, probs)])

        # Convert the frame back to a format suitable for OpenCV display (0-255 range, uint8)
        frame_display = (frm * 255).astype(np.uint8).copy()

        # Overlay prediction on frame
        cv2.putText(frame_display, f"Pred: {pred_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame_display, prob_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Live Prediction from Video', frame_display)
        
        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == "__main__":
    main()
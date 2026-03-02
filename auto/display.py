"""
Camera display with HUD overlay.

Shows the live camera feed in an OpenCV window with:
- State badge (AUTONOMOUS / RETURNING_HOME / MANUAL / AT_HOME)
- Action + confidence bar
- Position, heading, distance-to-home
- Command velocities (linear, angular)
- IMU correction status
- Mini-map with robot trail and home marker
- Time until next return

Runs on a configurable update rate (default 10 Hz) and is completely
optional — the controller works fine without a display attached.
"""

import math
import time
import threading
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

WINDOW_NAME = "JetRacer"

# Colour palette (BGR)
_COL = {
    'autonomous':     (0, 200, 0),
    'returning_home': (0, 165, 255),
    'at_home':        (255, 200, 0),
    'manual_override':(0, 0, 255),
    'white':  (255, 255, 255),
    'black':  (0, 0, 0),
    'grey':   (60, 60, 60),
    'green':  (0, 220, 0),
    'red':    (0, 0, 220),
    'yellow': (0, 255, 255),
    'cyan':   (255, 255, 0),
    'home':   (0, 100, 255),
    'trail':  (200, 200, 200),
}

# Action colours
_ACT_COL = {
    'left':     (255, 200, 0),
    'straight': (0, 220, 0),
    'right':    (0, 200, 255),
    'reverse':  (0, 0, 220),
}


class DisplayWindow:
    """Threaded OpenCV window with HUD overlay on the camera feed."""

    def __init__(self, config: dict, width: int = 640, height: int = 480):
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for the camera display")

        self.width = width
        self.height = height

        display_hz = config.get('inference', {}).get('display_rate', 10)
        self.interval = 1.0 / max(display_hz, 1)

        # Mini-map settings
        self.map_size = 140       # pixels
        self.map_scale = 30.0     # pixels per metre
        self.trail_max = 300      # max trail points

        # Shared state (written by controller, read by display)
        self._lock = threading.Lock()
        self._frame = None
        self._state = 'autonomous'
        self._action = 'straight'
        self._confidence = 0.0
        self._pos = (0.0, 0.0, 0.0)    # x, y, theta
        self._home = (0.0, 0.0)
        self._dist_home = 0.0
        self._linear = 0.0
        self._angular = 0.0
        self._imu_locked = False
        self._imu_yaw_err = 0.0
        self._time_to_return = 0.0
        self._trail = []          # list of (x, y) in metres

        self._running = False
        self._thread = None

    # ---- public API (called from controller) ----

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass

    def update(self, frame, *, state, action, confidence,
               pos, home, dist_home, linear, angular,
               imu_locked, imu_yaw_err, time_to_return):
        """Push new data from the control loop (non-blocking)."""
        with self._lock:
            self._frame = frame
            self._state = state
            self._action = action
            self._confidence = confidence
            self._pos = pos
            self._home = home
            self._dist_home = dist_home
            self._linear = linear
            self._angular = angular
            self._imu_locked = imu_locked
            self._imu_yaw_err = imu_yaw_err
            self._time_to_return = time_to_return
            self._trail.append((pos[0], pos[1]))
            if len(self._trail) > self.trail_max:
                self._trail = self._trail[-self.trail_max:]

    # ---- internal display thread ----

    def _display_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while self._running:
            t0 = time.time()

            with self._lock:
                frame = self._frame.copy() if self._frame is not None else None
                state = self._state
                action = self._action
                confidence = self._confidence
                pos = self._pos
                home = self._home
                dist_home = self._dist_home
                lin = self._linear
                ang = self._angular
                imu_locked = self._imu_locked
                imu_err = self._imu_yaw_err
                ttr = self._time_to_return
                trail = list(self._trail)

            if frame is None:
                # No frame yet — show placeholder
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COL['white'], 2)

            self._draw_hud(frame, state, action, confidence, pos, home,
                           dist_home, lin, ang, imu_locked, imu_err, ttr, trail)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False

            elapsed = time.time() - t0
            sleep = self.interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass

    # ---- drawing helpers ----

    def _draw_hud(self, frame, state, action, confidence, pos, home,
                  dist_home, lin, ang, imu_locked, imu_err, ttr, trail):
        h, w = frame.shape[:2]

        # 1) Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), _COL['black'], -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # State badge
        state_col = _COL.get(state, _COL['white'])
        state_label = state.upper().replace('_', ' ')
        cv2.putText(frame, state_label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_col, 2)

        # Action + confidence
        act_col = _ACT_COL.get(action, _COL['white'])
        cv2.putText(frame, f"{action.upper()}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, act_col, 2)
        # Confidence bar
        bar_x = 140
        bar_w = 120
        cv2.rectangle(frame, (bar_x, 36), (bar_x + bar_w, 50), _COL['grey'], -1)
        fill = int(bar_w * min(confidence, 1.0))
        cv2.rectangle(frame, (bar_x, 36), (bar_x + fill, 50), act_col, -1)
        cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 5, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COL['white'], 1)

        # Return timer (top-right)
        min_r = int(ttr) // 60
        sec_r = int(ttr) % 60
        ret_str = f"Home in {min_r}:{sec_r:02d}"
        cv2.putText(frame, ret_str, (w - 180, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL['cyan'], 1)

        # 2) Bottom info bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), _COL['black'], -1)
        cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

        x, y, th = pos
        deg = math.degrees(th)
        info1 = f"Pos:({x:+.2f},{y:+.2f}) Hdg:{deg:+.0f}d"
        cv2.putText(frame, info1, (10, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COL['white'], 1)

        info2 = f"Cmd: L={lin:+.2f} A={ang:+.2f}"
        cv2.putText(frame, info2, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COL['green'], 1)

        hx, hy = home
        info3 = f"Home:({hx:+.2f},{hy:+.2f}) Dist:{dist_home:.2f}m"
        cv2.putText(frame, info3, (300, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COL['home'], 1)

        imu_str = f"IMU err:{imu_err:+.1f}d" if imu_locked else "IMU: free"
        imu_col = _COL['green'] if imu_locked else _COL['yellow']
        cv2.putText(frame, imu_str, (300, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, imu_col, 1)

        # 3) Direction arrow (centre of frame)
        self._draw_arrow(frame, action, w // 2, h // 2)

        # 4) Mini-map (bottom-right)
        self._draw_minimap(frame, pos, home, trail, w, h)

    def _draw_arrow(self, frame, action, cx, cy):
        """Small directional arrow indicating the current action."""
        size = 28
        col = _ACT_COL.get(action, _COL['white'])
        thickness = 2
        if action == 'straight':
            cv2.arrowedLine(frame, (cx, cy + size), (cx, cy - size), col, thickness, tipLength=0.35)
        elif action == 'left':
            cv2.arrowedLine(frame, (cx + size, cy), (cx - size, cy), col, thickness, tipLength=0.35)
        elif action == 'right':
            cv2.arrowedLine(frame, (cx - size, cy), (cx + size, cy), col, thickness, tipLength=0.35)
        elif action == 'reverse':
            cv2.arrowedLine(frame, (cx, cy - size), (cx, cy + size), col, thickness, tipLength=0.35)

    def _draw_minimap(self, frame, pos, home, trail, fw, fh):
        """Draw a small top-down map in the bottom-right corner."""
        ms = self.map_size
        margin = 8
        # Origin of mini-map (top-left)
        ox = fw - ms - margin
        oy = fh - ms - margin - 50   # above bottom bar

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + ms, oy + ms), _COL['black'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (ox, oy), (ox + ms, oy + ms), _COL['grey'], 1)

        cx = ox + ms // 2
        cy = oy + ms // 2
        rx, ry, rth = pos
        hx, hy = home

        def world_to_map(wx, wy):
            # Centre map on robot
            dx = (wx - rx) * self.map_scale
            dy = -(wy - ry) * self.map_scale   # flip y
            mx = int(cx + dx)
            my = int(cy + dy)
            # Clamp inside map
            mx = max(ox + 1, min(ox + ms - 1, mx))
            my = max(oy + 1, min(oy + ms - 1, my))
            return mx, my

        # Trail
        if len(trail) > 1:
            for i in range(1, len(trail)):
                p1 = world_to_map(trail[i - 1][0], trail[i - 1][1])
                p2 = world_to_map(trail[i][0], trail[i][1])
                cv2.line(frame, p1, p2, _COL['trail'], 1)

        # Home marker
        hmx, hmy = world_to_map(hx, hy)
        cv2.drawMarker(frame, (hmx, hmy), _COL['home'], cv2.MARKER_DIAMOND, 10, 2)

        # Robot dot + heading line
        cv2.circle(frame, (cx, cy), 4, _COL['green'], -1)
        ex = int(cx + 14 * math.cos(rth))
        ey = int(cy - 14 * math.sin(rth))
        cv2.line(frame, (cx, cy), (ex, ey), _COL['green'], 2)

        # Scale label
        cv2.putText(frame, "map", (ox + 4, oy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, _COL['grey'], 1)

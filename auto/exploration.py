"""
Simple exploration strategy for autonomous mode.

Instead of driving purely reactively (just following vision predictions),
the explorer generates frontier waypoints so the robot systematically
covers new ground.  It keeps a coarse occupancy grid of *visited* cells
and picks the nearest unvisited frontier as the next exploration target.

When a frontier waypoint is active the robot biases its heading towards
that waypoint while still obeying vision-based obstacle avoidance.
If vision says "turn" or "reverse" (obstacle), the explorer yields.
When vision says "straight" and the robot is not heading toward the
waypoint, it nudges angular velocity to steer toward it.

This is intentionally lightweight — no SLAM, no lidar, just visited-cell
tracking from dead-reckoning + a simple greedy frontier picker.
"""

import math
import random
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

# Grid resolution in metres
CELL_SIZE = 0.4

# How far ahead to look for frontiers (metres)
MAX_FRONTIER_DIST = 3.0
MIN_FRONTIER_DIST = 0.6

# Angular nudge added when steering toward a frontier (rad/s)
EXPLORE_ANGULAR_NUDGE = 0.35


@dataclass
class ExplorationTarget:
    x: float
    y: float
    bearing: float = 0.0  # radians relative to robot heading


class ExplorationStrategy:
    """Lightweight visited-cell exploration planner."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        exp_cfg = config.get('exploration', {})
        self.cell_size = exp_cfg.get('cell_size', CELL_SIZE)
        self.max_dist = exp_cfg.get('max_frontier_dist', MAX_FRONTIER_DIST)
        self.min_dist = exp_cfg.get('min_frontier_dist', MIN_FRONTIER_DIST)
        self.nudge = exp_cfg.get('angular_nudge', EXPLORE_ANGULAR_NUDGE)

        # Visited cells: set of (gx, gy)
        self._visited: set = set()

        # Current exploration target
        self._target: Optional[ExplorationTarget] = None

        # Target tolerance
        self._target_tol = self.cell_size * 1.2

    # ---- public API ----

    def record_position(self, x: float, y: float):
        """Mark the cell at (x, y) as visited."""
        gx, gy = self._to_grid(x, y)
        self._visited.add((gx, gy))

    def get_target(self) -> Optional[ExplorationTarget]:
        """Return the current exploration target, or None."""
        return self._target

    def update(self, x: float, y: float, theta: float) -> Optional[ExplorationTarget]:
        """
        Update exploration state given current robot pose.
        Returns the active target (may be new or existing).
        """
        self.record_position(x, y)

        # Check if we've reached the current target
        if self._target is not None:
            dx = self._target.x - x
            dy = self._target.y - y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self._target_tol:
                self.logger.info(f"Reached frontier ({self._target.x:.1f},{self._target.y:.1f})")
                self._target = None

        # Pick new target if needed
        if self._target is None:
            self._target = self._pick_frontier(x, y, theta)
            if self._target is not None:
                self.logger.info(
                    f"New frontier target: ({self._target.x:.1f},{self._target.y:.1f}) "
                    f"bearing {math.degrees(self._target.bearing):+.0f}d")

        # Update bearing to target
        if self._target is not None:
            dx = self._target.x - x
            dy = self._target.y - y
            target_angle = math.atan2(dy, dx)
            self._target.bearing = self._normalize(target_angle - theta)

        return self._target

    def compute_angular_bias(self, vision_action: str, vision_confidence: float,
                             linear: float, angular: float) -> float:
        """
        Return a modified angular velocity that steers toward the exploration
        target, but only when vision says 'straight' (no obstacle).

        If vision detects an obstacle (left/right/reverse with high confidence),
        the exploration bias is suppressed and the vision command is trusted.
        """
        if self._target is None:
            return angular

        # Don't override obstacle avoidance
        if vision_action in ('left', 'right', 'reverse') and vision_confidence > 0.4:
            return angular

        # Only nudge when basically going straight
        if abs(angular) > 0.3:
            return angular

        bearing = self._target.bearing
        # Dead-zone: if already heading close to target, no nudge
        if abs(bearing) < 0.15:
            return angular

        # Proportional nudge toward target, clamped
        nudge = max(-self.nudge, min(self.nudge, bearing * 0.6))
        return angular + nudge

    def clear(self):
        """Reset visited cells and target (e.g. after home reset)."""
        self._visited.clear()
        self._target = None

    @property
    def cells_visited(self) -> int:
        return len(self._visited)

    # ---- internals ----

    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def _pick_frontier(self, x: float, y: float, theta: float) -> Optional[ExplorationTarget]:
        """Find the best unvisited cell within range."""
        gx, gy = self._to_grid(x, y)
        radius_cells = int(math.ceil(self.max_dist / self.cell_size))

        candidates = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cx, cy = gx + dx, gy + dy
                if (cx, cy) in self._visited:
                    continue
                # World coords of cell centre
                wx = (cx + 0.5) * self.cell_size
                wy = (cy + 0.5) * self.cell_size
                dist = math.sqrt((wx - x) ** 2 + (wy - y) ** 2)
                if dist < self.min_dist or dist > self.max_dist:
                    continue
                # Prefer cells roughly in front of the robot
                angle_to = math.atan2(wy - y, wx - x)
                bearing = abs(self._normalize(angle_to - theta))
                # Score: prefer close + in-front
                score = dist + bearing * 0.5
                candidates.append((score, wx, wy, self._normalize(angle_to - theta)))

        if not candidates:
            # All nearby cells visited — pick a random direction to push outward
            angle = theta + random.uniform(-0.8, 0.8)
            d = self.max_dist * 0.8
            wx = x + d * math.cos(angle)
            wy = y + d * math.sin(angle)
            return ExplorationTarget(x=wx, y=wy, bearing=self._normalize(angle - theta))

        candidates.sort(key=lambda c: c[0])
        _, bx, by, bb = candidates[0]
        return ExplorationTarget(x=bx, y=by, bearing=bb)

    @staticmethod
    def _normalize(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

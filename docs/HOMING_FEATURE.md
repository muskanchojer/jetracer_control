# Homing & Return-to-Start Feature

This document describes the autonomous homing feature that allows the robot to return to its starting position every N minutes with dead reckoning and stuck recovery capabilities.

## Overview

The homing feature provides:

1. **Autonomous AI Vision Navigation** - Uses the same vision policy as Autonomous mode to navigate
2. **Dead Reckoning Position Tracking** - Tracks robot position by integrating commanded velocities
3. **Automatic Return to Start** - Returns home every N minutes (configurable, default: 2 min)
4. **Stuck Detection** - Detects when the robot is stuck and not making progress
5. **Intelligent Recovery** - Reverses recent commands to escape stuck situations

## How Movement Works

**The robot is NOT moving mindlessly!** In Homing mode, the robot:

1. Uses the **same AI vision policy** as Autonomous mode to decide actions (left, right, straight, reverse)
2. Navigates intelligently, avoiding obstacles using the trained neural network
3. **Additionally** tracks its position using dead reckoning
4. Returns to start every N minutes based on time interval

So the movement is identical to Autonomous mode - camera → AI model → steering decision. The difference is that Homing mode adds position tracking and periodic return-to-start.

## Quick Start

### From Main Menu (Recommended)

```bash
python main.py
# Select option 4: Homing Mode
```

### Command Line

```bash
# Direct mode selection
python main.py --mode homing

# Or run standalone script
python scripts/homing_control.py

# Custom return interval (e.g., every 5 minutes)
python scripts/homing_control.py --return-interval 300

# Disable stuck detection
python scripts/homing_control.py --no-stuck-detection
```

### Keyboard Controls

While running:
- **Q** - Quit the program
- **H** - Force an immediate return home

## How It Works

### Dead Reckoning

The system tracks position using dead reckoning - it integrates the velocity commands sent to the motors to estimate position:

```
Position Update:
  dx = linear_velocity * cos(theta) * dt
  dy = linear_velocity * sin(theta) * dt
  dtheta = angular_velocity * dt
```

For arc motion (when turning), it uses a more accurate model:
```
  r = linear_velocity / angular_velocity  (radius of curvature)
  dx = r * (sin(theta_new) - sin(theta_old))
  dy = -r * (cos(theta_new) - cos(theta_old))
```

### Return Home Algorithm

When it's time to return home:

1. Calculate vector from current position to origin (0, 0)
2. Compute desired heading toward home
3. If heading error is large, rotate in place first
4. Move forward with proportional heading correction
5. Repeat until within tolerance of home

### Stuck Detection

The system detects when the robot is stuck by:

1. Recording position history every second
2. Checking if distance traveled over recent time window is below threshold
3. After N consecutive "stuck" detections, triggers recovery mode

### Stuck Recovery

When stuck is detected:

1. **Simple Recovery**: Reverse recent commands in opposite order
   - If robot moved forward + left, now move backward + right
   
2. **Intelligent Escape**: If simple reversal doesn't work:
   - Analyze command history patterns
   - Back up at opposite average velocity
   - Turn in opposite direction of recent tendency
   - Attempt forward motion in new direction

## Configuration

Edit `config/autonomous_config.yaml`:

```yaml
homing:
  # Return timing
  return_interval_seconds: 120  # Every 2 minutes
  home_tolerance: 0.3           # Within 30cm of start
  heading_tolerance: 0.2        # ~11 degrees
  
  # Return speeds
  return_speed: 0.3             # m/s forward
  turn_speed: 0.8               # rad/s turning
  
  # Dead reckoning
  dead_reckoning:
    enabled: true
    wheel_base: 0.20            # Robot wheel base in meters
    position_decay: 0.999       # Drift reduction factor
  
  # Stuck detection
  stuck_detection:
    enabled: true
    threshold_distance: 0.05    # Must move 5cm minimum
    check_interval: 1.0         # Check every second
    frames_threshold: 5         # 5 consecutive stuck = trigger
    max_recovery_commands: 10   # Reverse last 10 commands
  
  # Recovery maneuvers
  reversal:
    reverse_speed: 0.2
    turn_speed: 0.6
    reverse_duration: 1.0
    turn_duration: 0.5
```

## Architecture

### Components

```
HomingController (auto/homing_controller.py)
├── Dead Reckoning State
│   ├── Position (x, y, theta)
│   └── Total distance traveled
├── Command History
│   └── Ring buffer of last ~1000 commands
├── Stuck Detector
│   └── Position history analysis
└── Mode State Machine
    ├── EXPLORING - Normal driving
    ├── RETURNING_HOME - Going back to start
    ├── AT_HOME - At starting position
    └── STUCK_RECOVERY - Executing escape

ReversalPlanner (auto/homing_controller.py)
└── Plans escape sequences from command history

HomingAutonomousController (scripts/homing_control.py)
├── Integrates with vision/policy pipeline
├── Manages mode transitions
└── Executes homing/recovery maneuvers
```

### Data Flow

```
Vision → Policy → Action → ControlAdapter
                     ↓
              HomingController
              ├── Record command
              ├── Update dead reckoning
              └── Check stuck/return
```

## API Reference

### HomingController

```python
from auto.homing_controller import HomingController, HomingMode

# Create controller
homing = HomingController(config)

# Start tracking session
homing.start_session()

# Record each command for dead reckoning
homing.record_command(linear_vel, angular_vel, duration)

# Check if should return home
if homing.check_should_return_home():
    homing.start_return_home()
    while homing.get_mode() == HomingMode.RETURNING_HOME:
        linear, angular = homing.get_return_home_command()
        send_velocity(linear, angular)

# Check if stuck
if homing.is_stuck():
    homing.start_stuck_recovery()
    while cmd := homing.get_next_recovery_command():
        send_velocity(cmd.linear_velocity, cmd.angular_velocity)

# Get current position
x, y, theta = homing.get_position()

# Get comprehensive status
status = homing.get_status()
```

### HomingMode Enum

```python
HomingMode.EXPLORING       # Normal autonomous movement
HomingMode.RETURNING_HOME  # Actively returning to start
HomingMode.AT_HOME         # At starting position
HomingMode.STUCK_RECOVERY  # Executing recovery maneuvers
```

## Limitations

1. **Dead Reckoning Drift**: Position estimate will drift over time due to:
   - Wheel slip
   - Uneven terrain
   - Motor calibration differences
   
   Mitigation: Use `position_decay` factor and periodic sensor fusion with odometry

2. **Complex Stuck Situations**: Simple reversal may not work for:
   - Wedged between obstacles
   - High-friction surfaces
   
   Mitigation: Limit exploration area, use vision for obstacle detection

3. **No Absolute Positioning**: System relies on relative position from start
   
   Mitigation: Can fuse with ROS odometry using `fuse_with_external_position()`

## Tips

- **Indoor Use**: Works best on flat, indoor surfaces with good traction
- **Return Interval**: Shorter intervals = less drift accumulation
- **Stuck Recovery**: If frequently stuck, adjust reversal parameters
- **Monitoring**: Watch the camera window for position/status overlay

## Troubleshooting

**Robot doesn't return home accurately**
- Increase return frequency (shorter interval)
- Calibrate wheel_base parameter
- Enable sensor fusion with ROS odometry

**Frequently triggers stuck detection**
- Increase `threshold_distance` 
- Increase `frames_threshold`
- Check for mechanical issues

**Recovery doesn't escape stuck situation**
- Increase `max_recovery_commands`
- Adjust reversal speeds
- Consider manual intervention for complex obstacles

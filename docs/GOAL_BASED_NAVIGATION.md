# Goal-Based Navigation System

## Overview
The hybrid navigation system combines high-level waypoint navigation with low-level vision-based obstacle avoidance, giving your JetRacer goal-directed behavior while maintaining safety.

## Architecture

```
┌────────────────────────────────────────────────┐
│  User Input: Set Goals/Waypoints               │
└────────────┬───────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  WaypointNavigator (auto/navigation.py)        │
│  - Subscribes to /odom, /imu (ROS1 topics)    │
│  - Tracks current position                     │
│  - Calculates heading & distance to goal       │
│  - Outputs: NavigationCommand                  │
└────────────┬───────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────┐
│  FusionController (auto/fusion_controller.py)  │
│  - Blends navigation + vision commands         │
│  - Priority: Vision (safety) > Navigation      │
│  - Outputs: Motor commands (linear, angular)   │
└────────┬──────────┬────────────────────────────┘
         ↓          ↓
    Navigation   Vision
      System     System
         ↓          ↓
┌────────────────────────────────────────────────┐
│  ROS1 /cmd_vel topic → Robot motors            │
└────────────────────────────────────────────────┘
```

## Components

### 1. **WaypointNavigator** (`auto/navigation.py`)
- **Purpose**: High-level goal-directed navigation
- **ROS Topics**:
  - Subscribes to `/odom` (nav_msgs/Odometry) - position tracking
  - Subscribes to `/imu` (sensor_msgs/Imu) - orientation backup
- **Features**:
  - Single goal or multiple waypoints
  - Automatic waypoint progression
  - Distance-based slowdown near goals
  - Heading-based speed modulation

### 2. **FusionController** (`auto/fusion_controller.py`)
- **Purpose**: Blend navigation and vision commands
- **Strategy**:
  - **Critical obstacles** → Vision overrides 100%
  - **Obstacles detected** → Vision 80%, Navigation 20%
  - **Clear path** → Navigation 70%, Vision 30%
- **Safety-first**: Vision always has priority for obstacle avoidance

### 3. **GoalBasedController** (`scripts/goal_based_control.py`)
- **Purpose**: Main entry point for goal-based navigation
- **Combines**: Vision system + Navigation system + Fusion logic

## Usage

### Method 1: Interactive Mode
```bash
python mainv1_1.py
# Select option 3: Goal-Based (Navigation + Vision)
# Enter goal coordinates when prompted
```

### Method 2: Command Line
```bash
# Single goal
python scripts/goal_based_control.py --goal 2.0,1.5

# Multiple waypoints
python scripts/goal_based_control.py --waypoints "1.0,0.0;2.0,1.0;2.0,2.0"

# Direct mode selection
python mainv1_1.py --mode goal_based --goal 3.0,0.0
```

### Method 3: Programmatic (Python Code)
```python
from scripts.goal_based_control import GoalBasedController

controller = GoalBasedController()

# Set single goal
controller.set_goal(x=2.0, y=1.5)  # 2m forward, 1.5m left

# Set multiple waypoints
controller.set_waypoints([
    (1.0, 0.0),  # First waypoint
    (2.0, 1.0),  # Second waypoint
    (2.0, 2.0),  # Final waypoint
])

# Run
controller.run()
```

## Configuration

Edit `config/autonomous_config.yaml`:

```yaml
# Navigation parameters
navigation:
  max_speed: 0.4              # Maximum speed (m/s)
  min_speed: 0.1              # Minimum speed (m/s)
  max_angular_speed: 1.5      # Max turning speed (rad/s)
  waypoint_tolerance: 0.3     # Distance to reach goal (m)
  slowdown_distance: 1.0      # Start slowing at 1m from goal

# Fusion parameters
fusion:
  vision_weight: 0.7          # Vision priority (0-1)
  navigation_weight: 0.3      # Navigation priority (0-1)
  obstacle_confidence_threshold: 0.6
  critical_obstacle_threshold: 0.8
```

## How It Works

### Example: Navigate to (2m, 1m)

1. **Set Goal**: `controller.set_goal(2.0, 1.0)`

2. **Navigation Loop**:
   - Subscribe to `/odom` → Get current position (0, 0)
   - Calculate: distance=2.24m, bearing=26.6°
   - Output: "Move forward 0.4 m/s, turn toward 26.6°"

3. **Vision Processing**:
   - Camera sees obstacle on right
   - YOLOv8 detects objects
   - TinyNet predicts: "turn left" (confidence: 0.75)

4. **Fusion**:
   - Obstacle detected (confidence > 0.6)
   - Mode: "Vision Priority Blend"
   - Command: 80% vision (left turn) + 20% navigation (toward goal)
   - Result: Turn left to avoid obstacle while trending toward goal

5. **Execution**:
   - Publish to `/cmd_vel`: linear=0.3, angular=0.6
   - Robot motors receive commands via ROS

6. **Progress**:
   - As robot moves, odometry updates position
   - When distance < 0.3m → Goal reached!
   - Move to next waypoint (if any)

## ROS1 Compatibility

✅ **Fully compatible with ROS1** (Melodic/Noetic)

The system uses standard ROS1 topics that your Docker container already provides:
- `/cmd_vel` - geometry_msgs/Twist (motor control)
- `/odom` - nav_msgs/Odometry (position tracking)
- `/imu` - sensor_msgs/Imu (orientation)

No changes needed to the Docker container or jetracer_ros package!

## Testing

### Test 1: Simple Forward Goal
```bash
python scripts/goal_based_control.py --goal 1.0,0.0
# Robot should drive 1 meter forward, avoiding obstacles
```

### Test 2: Square Pattern
```bash
python scripts/goal_based_control.py --waypoints "1,0;1,1;0,1;0,0"
# Robot should drive in a square
```

### Test 3: Vision-Only (No Goal)
```bash
python scripts/goal_based_control.py
# Press Enter at goal prompt
# Robot uses vision for obstacle avoidance only
```

## Troubleshooting

**Odometry not updating?**
- Check Docker container is running: `docker ps`
- Check ROS topics: `docker exec jetracer_ros bash -c "source /root/catkin_ws/devel/setup.bash && rostopic list"`
- Verify robot base is running: `rostopic echo /odom`

**Robot not moving toward goal?**
- Check navigation parameters in config
- Increase `navigation_weight` in fusion config
- Verify goal coordinates are reasonable (< 5m)

**Obstacles ignored?**
- Decrease `obstacle_confidence_threshold`
- Check vision system is working (watch camera window)
- Increase `vision_weight`

## Key Features

✅ **ROS1 Integration**: Uses standard topics from waveshare jetracer_ros  
✅ **Safety-First**: Vision always overrides for obstacle avoidance  
✅ **Multi-Waypoint**: Sequential waypoint following  
✅ **Smooth Blending**: Gradual transitions between navigation and vision control  
✅ **Distance-based slowdown**: Reduces speed near goals  
✅ **Real-time odometry**: Tracks position using /odom topic  

## Next Steps

1. **Test basic navigation**: Try simple forward goals first
2. **Tune parameters**: Adjust speeds and weights in config
3. **Add RViz visualization**: Visualize path and obstacles
4. **Implement path planning**: Add A* or other planners for complex environments
5. **Save trajectories**: Record successful paths for replay

#!/usr/bin/env python3
"""
JetRacer Unified Control System
Main entry point with mode selection: Manual (keyboard) or Autonomous (AI)
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import mainv1 
from mainv1 import JetRacerIntegrated

def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           JetRacer Autonomous Control System              ║
║                                                           ║
║  🚗  Manual Mode      - Keyboard control (WASD)          ║
║  🤖  Autonomous Mode  - AI vision (free roam)            ║
║  🎯  Goal-Based Mode  - Navigation + Vision fusion       ║
║  🏠  Homing Mode      - Auto return to start + recovery  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def select_mode():
    """Interactive mode selection"""
    print("\nSelect control mode:")
    print("  1. Manual (Keyboard Control)")
    print("  2. Autonomous (AI Vision - Free Roam)")
    print("  3. Goal-Based (Navigation + Vision)")
    print("  4. Homing (Autonomous + Return to Start)")
    print("  5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                return 'manual'
            elif choice == '2':
                return 'autonomous'
            elif choice == '3':
                return 'goal_based'
            elif choice == '4':
                return 'homing'
            elif choice == '5':
                return 'exit'
            else:
                print("Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return 'exit'
        except EOFError:
            return 'exit'

def check_docker():
    """Check if Docker container is running"""
    import subprocess
    
    container = "jetracer_ros"
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        
        if container in result.stdout:
            print(f"✅ Docker container '{container}' is running")
            return True
        else:
            print(f"\n❌ Docker container '{container}' is not running")
            print(f"Start it with: ~/start_jetracer_docker.sh\n")
            return False
    
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def run_manual_mode():
    """Run manual keyboard control mode"""
    print("\n" + "="*60)
    print("Starting Manual Control Mode (Keyboard)")
    print("="*60 + "\n")
    
    # Import and run keyboard control from main_old.py
    try:
        # Use the existing keyboard control implementation
        
        robot = JetRacerIntegrated()
        
        if not robot.check_docker():
            print("Please start Docker first")
            return False
        
        if not robot.check_hardware():
            print("Hardware check failed")
            return False
        
        robot.sync_scripts()
        
        # Ask if user wants camera window
        print("\nShow camera window? (y/n): ", end='', flush=True)
        show_camera = input().strip().lower() == 'y'
        
        # Start keyboard control (this will start roscore and robot_base first)
        print("\n🎮 Keyboard control active")
        print("Controls: W=forward, A=left, S=backward, D=right, Space=stop, Q=quit")
        
        if show_camera:
            print("📷 Camera window enabled\n")
            # Start with camera
            robot.run()  # Full integration mode
        else:
            print("\n")
            # Keyboard only
            robot.start_keyboard_control()
        
    except ImportError as e:
        print(f"❌ Failed to import keyboard control: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in manual mode: {e}")
        return False
    
    return True

def run_autonomous_mode():
    """Run autonomous AI navigation mode (free roam - no goal)"""
    print("\n" + "="*60)
    print("Starting Autonomous Mode (AI Vision - Free Roam)")
    print("="*60 + "\n")
    
    try:
        from scripts.autonomous_control import AutonomousController
        
        controller = AutonomousController()
        return controller.run()
    
    except ImportError as e:
        print(f"❌ Failed to import autonomous controller: {e}")
        print("\nMake sure required packages are installed:")
        print("  pip install ultralytics opencv-python numpy pyyaml")
        return False
    except Exception as e:
        print(f"❌ Error in autonomous mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_goal_based_mode():
    """Run goal-based navigation mode (navigation + vision fusion)"""
    print("\n" + "="*60)
    print("Starting Goal-Based Mode (Navigation + Vision)")
    print("="*60 + "\n")
    
    try:
        from scripts.goal_based_control import GoalBasedController
        
        # Get goal from user
        print("Set navigation goal:")
        print("  Example: 2.0,1.5 means 2 meters forward, 1.5 meters left")
        print("  Or press Enter to set goal programmatically in code")
        
        goal_input = input("\nEnter goal (x,y in meters) or press Enter: ").strip()
        
        controller = GoalBasedController()
        
        if goal_input:
            try:
                x, y = map(float, goal_input.split(','))
                controller.set_goal(x, y)
            except ValueError:
                print("❌ Invalid goal format. Using vision-only mode.")
        else:
            print("\n💡 No goal set. You can set goals programmatically:")
            print("   controller.set_goal(x, y)")
            print("   controller.set_waypoints([(x1,y1), (x2,y2), ...])")
            print("\nStarting in vision-only mode (free roam)...")
        
        return controller.run()
    
    except ImportError as e:
        print(f"❌ Failed to import goal-based controller: {e}")
        print("\nMake sure required packages are installed:")
        print("  pip install ultralytics opencv-python numpy pyyaml")
        return False
    except Exception as e:
        print(f"❌ Error in goal-based mode: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_homing_mode():
    """Run homing mode (autonomous + return to start every N minutes)"""
    print("\n" + "="*60)
    print("Starting Homing Mode (Autonomous + Return to Start)")
    print("="*60 + "\n")
    
    print("📍 Homing Mode Features:")
    print("   - Autonomous AI vision navigation (same as Autonomous mode)")
    print("   - Tracks position using dead reckoning")
    print("   - Returns to start every N minutes (configurable)")
    print("   - Detects when stuck and auto-recovers")
    print("   - Press 'H' anytime to force return home\n")
    
    try:
        # Load config to get default interval
        import yaml
        config_path = Path(__file__).parent / "config" / "autonomous_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        default_interval = config.get('homing', {}).get('return_interval_seconds', 120)
        
        # Ask for return interval
        print(f"Return interval (seconds)? [default: {default_interval}]")
        interval_input = input("Enter interval or press Enter for default: ").strip()
        
        if interval_input:
            try:
                return_interval = int(interval_input)
            except ValueError:
                print(f"Invalid number, using default: {default_interval}s")
                return_interval = default_interval
        else:
            return_interval = default_interval
        
        # Ask about stuck detection
        print("\nEnable stuck detection and auto-recovery? [Y/n]: ", end='')
        stuck_input = input().strip().lower()
        enable_stuck = stuck_input != 'n'
        
        print(f"\n🏠 Will return home every {return_interval/60:.1f} minutes")
        print(f"🔧 Stuck recovery: {'enabled' if enable_stuck else 'disabled'}\n")
        
        from scripts.homing_control import HomingAutonomousController
        
        controller = HomingAutonomousController(
            return_interval=return_interval,
            enable_stuck_detection=enable_stuck
        )
        return controller.run()
    
    except ImportError as e:
        print(f"❌ Failed to import homing controller: {e}")
        print("\nMake sure required packages are installed:")
        print("  pip install ultralytics opencv-python numpy pyyaml")
        return False
    except Exception as e:
        print(f"❌ Error in homing mode: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JetRacer Control System')
    parser.add_argument('--mode', choices=['manual', 'autonomous', 'goal_based', 'homing'], 
                       help='Control mode (skip interactive selection)')
    parser.add_argument('--no-check', action='store_true',
                       help='Skip Docker check')
    parser.add_argument('--goal', type=str, 
                       help='Goal coordinates for goal_based mode (format: x,y)')
    parser.add_argument('--return-interval', type=int, default=120,
                       help='Return interval in seconds for homing mode (default: 120)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check Docker (unless skipped)
    if not args.no_check:
        if not check_docker():
            print("Cannot proceed without Docker running")
            return 1
    
    # Get mode (from args or interactive)
    if args.mode:
        mode = args.mode
        print(f"\nMode selected: {mode}")
    else:
        mode = select_mode()
    
    # Execute selected mode
    if mode == 'exit':
        print("Goodbye! 👋")
        return 0
    
    elif mode == 'manual':
        success = run_manual_mode()
        return 0 if success else 1
    
    elif mode == 'autonomous':
        success = run_autonomous_mode()
        return 0 if success else 1
    
    elif mode == 'goal_based':
        success = run_goal_based_mode()
        return 0 if success else 1
    
    elif mode == 'homing':
        success = run_homing_mode()
        return 0 if success else 1
    
    else:
        print(f"Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

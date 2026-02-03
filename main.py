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
import main_old 
from main_old import JetRacerIntegrated

def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           JetRacer Autonomous Control System              ║
║                                                           ║
║  🚗  Manual Mode     - Keyboard control (WASD)           ║
║  🤖  Autonomous Mode - AI-powered navigation             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def select_mode():
    """Interactive mode selection"""
    print("\nSelect control mode:")
    print("  1. Manual (Keyboard Control)")
    print("  2. Autonomous (AI Navigation)")
    print("  3. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                return 'manual'
            elif choice == '2':
                return 'autonomous'
            elif choice == '3':
                return 'exit'
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
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
    """Run autonomous AI navigation mode"""
    print("\n" + "="*60)
    print("Starting Autonomous Mode (AI Navigation)")
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

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JetRacer Control System')
    parser.add_argument('--mode', choices=['manual', 'autonomous'], 
                       help='Control mode (skip interactive selection)')
    parser.add_argument('--no-check', action='store_true',
                       help='Skip Docker check')
    
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

#!/usr/bin/env python3
"""
Test script for run logger integration with the runner.
This demonstrates how the logging system works without requiring an actual Android device.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.run_logger_integration import run_logging_session
from core.logging_config import get_logger

log = get_logger("TEST-RUNNER")

def test_runner_logging_integration():
    """Test the runner logging integration without actual Android automation."""
    
    goal = "Enable Wi-Fi in Android settings"
    serial = "emulator-5554"
    
    print("🧪 **Testing Run Logger Integration with Runner**")
    print("=" * 60)
    
    # Simulate the exact same flow that the runner would use
    with run_logging_session(goal) as run_logger:
        print(f"🚀 Starting automation with comprehensive logging...")
        print(f"📋 Goal: {goal}")
        print(f"📱 Device: {serial}")
        print(f"🆔 Run ID: {run_logger.run_id}")
        print(f"📁 Logs will be saved to: logs/run_{run_logger.run_id}/")
        print("=" * 60)
        
        try:
            # Log automation start (same as runner)
            run_logger.log_event("automation_start", "RUNNER", {
                "goal": goal,
                "device_serial": serial,
                "run_id": run_logger.run_id
            })
            
            # Simulate episode start (same as runner)
            episode_id = "episode_001"
            run_logger.log_episode_start(episode_id, goal)
            
            # Simulate some automation steps (what the agents would do)
            print("📱 Simulating Android automation steps...")
            
            # Step 1: Launch Settings
            step_1 = {"action": "launch_app", "package": "com.android.settings", "step_id": "step_001"}
            result_1 = {"success": True, "duration": 2.1}
            ui_1 = "<UI>Android Settings app launched</UI>"
            
            run_logger.log_step_execution(episode_id, step_1, result_1, ui_1)
            print("  ✅ Step 1: Settings app launched")
            
            # Step 2: Navigate to Wi-Fi settings
            step_2 = {"action": "tap", "resource_id": "wifi_settings", "step_id": "step_002"}
            result_2 = {"success": True, "duration": 1.5}
            ui_2 = "<UI>Wi-Fi settings screen visible</UI>"
            
            run_logger.log_step_execution(episode_id, step_2, result_2, ui_2)
            print("  ✅ Step 2: Navigated to Wi-Fi settings")
            
            # Step 3: Enable Wi-Fi toggle
            step_3 = {"action": "tap", "resource_id": "wifi_toggle", "step_id": "step_003"}
            result_3 = {"success": True, "duration": 0.8}
            ui_3 = "<UI>Wi-Fi toggle switched to ON</UI>"
            
            run_logger.log_step_execution(episode_id, step_3, result_3, ui_3)
            print("  ✅ Step 3: Wi-Fi enabled")
            
            # Simulate verification reports (what the verifier would do)
            print("🔍 Simulating verification reports...")
            
            verification_1 = {
                "verified": True,
                "confidence": 0.9,
                "reason": "Settings app successfully launched"
            }
            run_logger.log_verification_report(episode_id, step_1, verification_1, ui_1, None)
            print("  ✅ Verification 1: Settings app launch verified")
            
            verification_2 = {
                "verified": True,
                "confidence": 0.85,
                "reason": "Wi-Fi settings screen visible"
            }
            run_logger.log_verification_report(episode_id, step_2, verification_2, ui_2, None)
            print("  ✅ Verification 2: Wi-Fi settings navigation verified")
            
            verification_3 = {
                "verified": True,
                "confidence": 0.95,
                "reason": "Wi-Fi toggle shows ON state"
            }
            run_logger.log_verification_report(episode_id, step_3, verification_3, ui_3, None)
            print("  ✅ Verification 3: Wi-Fi enablement verified")
            
            # Log episode completion (same as runner)
            run_logger.log_episode_end(episode_id, "completed", "Wi-Fi successfully enabled in Android settings")
            
            # Log automation completion (same as runner)
            run_logger.log_event("automation_complete", "RUNNER", {
                "status": "success",
                "goal": goal,
                "episode_id": episode_id
            })
            
            print("=" * 60)
            print("✅ Automation completed successfully!")
            print(f"📊 Run log saved to: logs/run_{run_logger.run_id}/")
            
            return {"status": "success", "episode_id": episode_id, "steps": 3}
            
        except Exception as e:
            # Log automation failure (same as runner)
            run_logger.log_error("RUNNER", f"Automation failed: {str(e)}", context={
                "error_type": "automation_failure",
                "goal": goal,
                "device_serial": serial
            })
            
            print("=" * 60)
            print(f"❌ Automation failed: {e}")
            print(f"📊 Check the run log for detailed error information: logs/run_{run_logger.run_id}/")
            raise

def main():
    """Main test function."""
    try:
        result = test_runner_logging_integration()
        print(f"\n🎯 Test completed successfully!")
        print(f"📋 Result: {result}")
        print(f"\n💡 Now you can run the actual command:")
        print(f"   python -m runners.run_example --goal \"Enable Wi-Fi in Android settings\" --serial emulator-5554")
        print(f"\n📁 The logging JSON file will be automatically created and saved!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

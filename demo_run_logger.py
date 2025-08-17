#!/usr/bin/env python3
"""
Demo script for the Run Logger System

This script demonstrates how to use the comprehensive run logger
to capture all QA automation activities in a single JSON file per run.
"""

import uuid
import time
from pathlib import Path
from core.run_logger import RunLogger, set_run_logger, get_run_logger
from core.logging_config import get_logger

log = get_logger("DEMO-RUN-LOGGER")

def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("\n🔍 **Basic Logging Demo**")
    
    # Create a run logger
    run_id = str(uuid.uuid4())[:8]
    user_goal = "Launch weather app and check current conditions"
    
    print(f"Creating run logger for run: {run_id}")
    print(f"User goal: {user_goal}")
    
    run_logger = RunLogger(run_id, user_goal)
    set_run_logger(run_logger)
    
    # Log some events
    run_logger.log_event("demo_start", "DEMO", {"message": "Starting basic logging demo"})
    run_logger.log_event("user_input", "USER", {"goal": user_goal, "priority": "high"})
    
    # Simulate episode
    episode_id = "episode_001"
    run_logger.log_episode_start(episode_id, user_goal)
    
    # Simulate steps
    steps = [
        {"action": "launch_app", "package": "com.weather.app", "step_id": "step_001"},
        {"action": "tap", "resource_id": "search_button", "step_id": "step_002"},
        {"action": "type", "text": "New York", "step_id": "step_003"},
        {"action": "verify", "resource_id": "weather_results", "step_id": "step_004"}
    ]
    
    for step in steps:
        run_logger.log_step_execution(
            episode_id, 
            step, 
            {"success": True, "duration": 1.5}, 
            f"<UI>{step['action']} completed</UI>"
        )
        time.sleep(0.1)  # Simulate execution time
    
    # Simulate verification reports
    verification_results = [
        {"verified": True, "confidence": 0.9, "reason": "App launched successfully"},
        {"verified": True, "confidence": 0.85, "reason": "Search button found and tapped"},
        {"verified": True, "confidence": 0.95, "reason": "Text entered correctly"},
        {"verified": True, "confidence": 0.8, "reason": "Weather results displayed"}
    ]
    
    for i, result in enumerate(verification_results):
        run_logger.log_verification_report(
            episode_id,
            steps[i],
            result,
            f"<UI>Verification {i+1}</UI>",
            f"screenshot_{i+1}.png"
        )
    
    # End episode
    run_logger.log_episode_end(episode_id, "completed", "Weather app launched and search completed")
    
    # Save the run log
    log_file = run_logger.save_run_log()
    print(f"✅ Run log saved to: {log_file}")
    
    # Cleanup
    run_logger.cleanup()
    set_run_logger(None)
    
    return log_file

def demo_error_logging():
    """Demonstrate error logging functionality."""
    print("\n🚨 **Error Logging Demo**")
    
    # Create a run logger
    run_id = str(uuid.uuid4())[:8]
    user_goal = "Test error handling and logging"
    
    print(f"Creating run logger for run: {run_id}")
    
    run_logger = RunLogger(run_id, user_goal)
    set_run_logger(run_logger)
    
    # Log some errors
    run_logger.log_error("LLM-EXECUTOR", "Element not found", "episode_001", "step_001", 
                         {"ui_xml": "<UI>No search button found</UI>"})
    
    run_logger.log_error("LLM-VERIFIER", "Screenshot capture failed", "episode_001", "step_002",
                         {"error_type": "device_error", "retry_count": 3})
    
    # Log critical failure
    run_logger.log_critical_failure("episode_001", 
                                   {"action": "launch_app", "step_id": "step_001"},
                                   {"verified": False, "reason": "App not installed", "confidence": 0.0})
    
    # Save the run log
    log_file = run_logger.save_run_log()
    print(f"✅ Error run log saved to: {log_file}")
    
    # Cleanup
    run_logger.cleanup()
    set_run_logger(None)
    
    return log_file

def demo_multi_episode_logging():
    """Demonstrate multi-episode logging functionality."""
    print("\n🔄 **Multi-Episode Logging Demo**")
    
    # Create a run logger
    run_id = str(uuid.uuid4())[:8]
    user_goal = "Test multiple episodes with different outcomes"
    
    print(f"Creating run logger for run: {run_id}")
    
    run_logger = RunLogger(run_id, user_goal)
    set_run_logger(run_logger)
    
    # Episode 1: Successful
    episode_1 = "episode_001"
    run_logger.log_episode_start(episode_1, "Launch calculator app")
    
    run_logger.log_step_execution(episode_1, 
                                 {"action": "launch_app", "package": "com.calculator", "step_id": "step_001"},
                                 {"success": True, "duration": 2.1},
                                 "<UI>Calculator launched</UI>")
    
    run_logger.log_verification_report(episode_1,
                                     {"action": "launch_app", "step_id": "step_001"},
                                     {"verified": True, "confidence": 0.9, "reason": "Calculator app visible"},
                                     "<UI>Calculator interface displayed</UI>",
                                     "calc_launch.png")
    
    run_logger.log_episode_end(episode_1, "completed", "Calculator app launched successfully")
    
    # Episode 2: Failed
    episode_2 = "episode_002"
    run_logger.log_episode_start(episode_2, "Launch non-existent app")
    
    run_logger.log_step_execution(episode_2,
                                 {"action": "launch_app", "package": "com.nonexistent", "step_id": "step_001"},
                                 {"success": False, "error": "App not found", "duration": 1.0},
                                 "<UI>Error dialog shown</UI>")
    
    run_logger.log_error("LLM-EXECUTOR", "App not found", episode_2, "step_001",
                         {"package": "com.nonexistent", "available_apps": ["com.calculator", "com.weather"]})
    
    run_logger.log_episode_end(episode_2, "failed", "App not found on device")
    
    # Save the run log
    log_file = run_logger.save_run_log()
    print(f"✅ Multi-episode run log saved to: {log_file}")
    
    # Cleanup
    run_logger.cleanup()
    set_run_logger(None)
    
    return log_file

def analyze_run_log(log_file_path: str):
    """Analyze a saved run log file."""
    print(f"\n📊 **Analyzing Run Log: {log_file_path}**")
    
    try:
        import json
        with open(log_file_path, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
        
        metadata = run_data.get("metadata", {})
        print(f"Run ID: {metadata.get('run_id')}")
        print(f"User Goal: {metadata.get('user_goal')}")
        print(f"Duration: {metadata.get('duration', 0):.2f} seconds")
        print(f"Total Episodes: {metadata.get('total_episodes', 0)}")
        print(f"Successful Episodes: {metadata.get('successful_episodes', 0)}")
        print(f"Failed Episodes: {metadata.get('failed_episodes', 0)}")
        print(f"Total Steps: {metadata.get('total_steps', 0)}")
        print(f"Verification Success Rate: {metadata.get('verification_success_rate', 0):.2%}")
        print(f"Average Confidence: {metadata.get('average_confidence', 0):.2f}")
        print(f"Total Screenshots: {metadata.get('total_screenshots', 0)}")
        print(f"Total Errors: {metadata.get('total_errors', 0)}")
        
        # Show events summary
        events = run_data.get("events", [])
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print(f"\nEvent Summary:")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
        
        # Show episodes summary
        episodes = run_data.get("episodes", {})
        print(f"\nEpisode Summary:")
        for episode_id, episode_data in episodes.items():
            status = episode_data.get("status", "unknown")
            duration = episode_data.get("end_time", 0) - episode_data.get("start_time", 0)
            steps = episode_data.get("total_steps", 0)
            print(f"  {episode_id}: {status} ({duration:.2f}s, {steps} steps)")
        
    except Exception as e:
        print(f"❌ Failed to analyze run log: {e}")

def main():
    """Main demo function."""
    print("🚀 **Run Logger System Demo**")
    print("=" * 50)
    
    print("\nThis demo shows how the Run Logger captures all QA automation activities")
    print("and saves them to comprehensive JSON files for analysis and debugging.")
    
    # Run demos
    basic_log = demo_basic_logging()
    error_log = demo_error_logging()
    multi_episode_log = demo_multi_episode_logging()
    
    # Analyze the logs
    print("\n" + "=" * 50)
    print("📊 **LOG ANALYSIS**")
    print("=" * 50)
    
    analyze_run_log(basic_log)
    print("\n" + "-" * 30)
    analyze_run_log(error_log)
    print("\n" + "-" * 30)
    analyze_run_log(multi_episode_log)
    
    print("\n" + "=" * 50)
    print("🎯 **Demo Complete!**")
    print("=" * 50)
    print("\nThe Run Logger system provides:")
    print("✅ Complete audit trail of all automation activities")
    print("✅ Structured JSON output for easy analysis")
    print("✅ Screenshot and verification report tracking")
    print("✅ Error and failure logging with context")
    print("✅ Episode-level metrics and success rates")
    print("✅ System information and environment details")
    print("\nAll logs are saved to timestamped JSON files in the logs/ directory.")
    print("Each run gets its own subdirectory with complete execution data.")

if __name__ == "__main__":
    main()

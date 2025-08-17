# 🚀 **Run Logger System: Complete Usage Guide**

## 🎯 **Overview**

The **Run Logger System** is a comprehensive logging solution that captures **ALL QA automation activities** during a single run and saves them to a **single, timestamped JSON file**. This creates a complete audit trail that survives system restarts and provides comprehensive debugging capabilities.

## ✨ **Key Features**

- ✅ **Complete Audit Trail**: Every action, verification, and error is logged
- ✅ **Single JSON File**: All run data saved in one comprehensive file
- ✅ **Structured Data**: Easy to parse and analyze programmatically
- ✅ **Screenshot Tracking**: Links to all verification screenshots
- ✅ **Error Context**: Detailed error information with context
- ✅ **Performance Metrics**: Success rates, confidence scores, and timing
- ✅ **System Information**: Platform, environment, and configuration details
- ✅ **Episode Management**: Track multiple episodes within a single run

## 📁 **Output Structure**

```
logs/
├── run_[RUN_ID]/
│   ├── run_[RUN_ID]_[TIMESTAMP].json    # Complete run log
│   └── screenshots/                      # Run-specific screenshots
├── run_[RUN_ID]/
│   ├── run_[RUN_ID]_[TIMESTAMP].json    # Another run log
│   └── screenshots/
└── screenshots/                          # Global screenshots
```

## 🔧 **Basic Usage**

### **1. Simple Integration (Recommended)**

```python
from core.run_logger_integration import run_logging_session

# Use context manager for automatic cleanup
with run_logging_session("Launch weather app and check conditions") as run_logger:
    # Your automation code here
    episode_id = "episode_001"
    run_logger.log_episode_start(episode_id, "Launch weather app")
    
    # Log steps, verifications, errors, etc.
    run_logger.log_step_execution(episode_id, step, result, ui_xml)
    run_logger.log_verification_report(episode_id, step, result, ui_xml, screenshot_path)
    
    run_logger.log_episode_end(episode_id, "completed", "Goal achieved")

# Log file automatically saved and cleaned up
```

### **2. Manual Start/Stop**

```python
from core.run_logger_integration import start_run_logging, stop_run_logging

# Start logging
run_logger = start_run_logging("Test automation")

try:
    # Your automation code here
    run_logger.log_event("step_completed", "AGENT", {"step": "launch_app"})
    
finally:
    # Stop logging and save
    log_file = stop_run_logging()
    print(f"Log saved to: {log_file}")
```

### **3. Direct Integration with Agents**

The system automatically integrates with all agents when run logging is active:

```python
# Agents automatically log to run logger if available
# No code changes needed in existing automation scripts
```

## 📊 **What Gets Logged**

### **1. Run Metadata**
```json
{
  "metadata": {
    "run_id": "99cc5e7e",
    "user_goal": "Launch weather app and check conditions",
    "start_time": 1755406856.8314612,
    "end_time": 1755406857.5224948,
    "duration": 0.69,
    "total_episodes": 1,
    "successful_episodes": 1,
    "failed_episodes": 0,
    "total_steps": 4,
    "verification_success_rate": 1.0,
    "average_confidence": 0.875
  }
}
```

### **2. System Information**
```json
{
  "system_info": {
    "platform": "Windows-11-10.0.26200-SP0",
    "python_version": "3.12.2",
    "working_directory": "C:\\Users\\baliu\\Desktop\\multi_agent_qa",
    "environment_variables": {...}
  }
}
```

### **3. Complete Event Log**
```json
{
  "events": [
    {
      "timestamp": 1755406857.1063576,
      "event_type": "step_execution",
      "agent": "LLM-EXECUTOR",
      "episode_id": "episode_001",
      "step_id": "step_001",
      "action": "launch_app",
      "data": {
        "step": {"action": "launch_app", "package": "com.weather.app"},
        "result": {"success": true, "duration": 1.5},
        "ui_xml": "<UI>App launched</UI>"
      },
      "severity": "INFO"
    }
  ]
}
```

### **4. Episode Details**
```json
{
  "episodes": {
    "episode_001": {
      "episode_id": "episode_001",
      "user_goal": "Launch weather app",
      "start_time": 1755406857.1063576,
      "end_time": 1755406857.5156250,
      "status": "completed",
      "total_steps": 4,
      "successful_steps": 4,
      "failed_steps": 0,
      "steps": [...],
      "verification_reports": [...],
      "screenshots": [...]
    }
  }
}
```

### **5. Verification Reports**
```json
{
  "verification_reports": [
    {
      "step": {"action": "launch_app", "step_id": "step_001"},
      "result": {
        "verified": true,
        "confidence": 0.9,
        "reason": "App launched successfully"
      },
      "ui_xml": "<UI>Weather app visible</UI>",
      "screenshot_path": "screenshot_1.png",
      "timestamp": 1755406857.5117188
    }
  ]
}
```

### **6. Screenshots**
```json
{
  "screenshots": [
    {
      "episode_id": "episode_001",
      "step_id": "step_001",
      "screenshot_path": "screenshot_1.png",
      "description": "Verification screenshot",
      "timestamp": 1755406857.5117188
    }
  ]
}
```

### **7. Errors and Failures**
```json
{
  "errors": [
    {
      "error": "Element not found",
      "context": {"ui_xml": "<UI>No button found</UI>"},
      "timestamp": 1755406857.5362058
    }
  ]
}
```

## 🎮 **Advanced Usage**

### **1. Custom Event Logging**

```python
# Log custom events
run_logger.log_event(
    event_type="user_interaction",
    agent="USER",
    data={"action": "manual_override", "reason": "UI glitch"},
    episode_id="episode_001",
    step_id="step_003"
)
```

### **2. Error Logging with Context**

```python
# Log errors with rich context
run_logger.log_error(
    agent="LLM-EXECUTOR",
    error="Element not found",
    episode_id="episode_001",
    step_id="step_002",
    context={
        "ui_xml": "<UI>Current UI state</UI>",
        "expected_element": "search_button",
        "available_elements": ["menu_button", "back_button"]
    }
)
```

### **3. Critical Failure Logging**

```python
# Log critical failures that require manual intervention
run_logger.log_critical_failure(
    episode_id="episode_001",
    step={"action": "launch_app", "step_id": "step_001"},
    failure_result={
        "verified": false,
        "reason": "App not installed",
        "confidence": 0.0
    }
)
```

### **4. Screenshot Logging**

```python
# Log screenshots with descriptions
run_logger.log_screenshot(
    episode_id="episode_001",
    step_id="step_002",
    screenshot_path="screenshots/step_002.png",
    description="After tapping search button"
)
```

## 🔍 **Integration Examples**

### **1. Integration with Existing Runners**

```python
# runners/aitw_runner.py
from core.run_logger_integration import run_logging_session

def run_aitw_automation(user_goal: str):
    with run_logging_session(user_goal) as run_logger:
        # Existing automation code remains unchanged
        # All agent activities are automatically logged
        
        # Optional: Add custom logging
        run_logger.log_event("automation_start", "RUNNER", {"goal": user_goal})
        
        # Run your existing automation...
        result = execute_automation_pipeline(user_goal)
        
        run_logger.log_event("automation_complete", "RUNNER", {"result": result})
        
        return result
```

### **2. Integration with Test Scripts**

```python
# test_automation.py
from core.run_logger_integration import run_logging_session

def test_weather_app_automation():
    with run_logging_session("Test weather app automation") as run_logger:
        # Test setup
        run_logger.log_event("test_setup", "TEST", {"phase": "setup"})
        
        # Run tests
        test_result = run_weather_app_tests()
        
        # Log test results
        run_logger.log_event("test_complete", "TEST", {
            "result": test_result,
            "passed": test_result.get("passed", 0),
            "failed": test_result.get("failed", 0)
        })
        
        return test_result
```

### **3. Integration with Pipeline Executors**

```python
# full_pipeline_executor.py
from core.run_logger_integration import run_logging_session

def execute_full_pipeline(user_goal: str):
    with run_logging_session(user_goal) as run_logger:
        # Log pipeline start
        run_logger.log_event("pipeline_start", "PIPELINE", {"goal": user_goal})
        
        # Execute pipeline steps
        for step in pipeline_steps:
            run_logger.log_event("pipeline_step", "PIPELINE", {
                "step": step,
                "status": "starting"
            })
            
            # Execute step...
            result = execute_step(step)
            
            run_logger.log_event("pipeline_step", "PIPELINE", {
                "step": step,
                "status": "completed",
                "result": result
            })
        
        # Log pipeline completion
        run_logger.log_event("pipeline_complete", "PIPELINE", {"status": "success"})
```

## 📈 **Analysis and Reporting**

### **1. Programmatic Analysis**

```python
import json

def analyze_run_log(log_file_path: str):
    with open(log_file_path, 'r') as f:
        run_data = json.load(f)
    
    metadata = run_data["metadata"]
    print(f"Run Duration: {metadata['duration']:.2f}s")
    print(f"Success Rate: {metadata['verification_success_rate']:.1%}")
    print(f"Average Confidence: {metadata['average_confidence']:.2f}")
    
    # Analyze events
    events = run_data["events"]
    event_counts = {}
    for event in events:
        event_type = event["event_type"]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print(f"Event Summary: {event_counts}")
```

### **2. Performance Metrics**

```python
def get_performance_metrics(run_data):
    metadata = run_data["metadata"]
    
    return {
        "efficiency": metadata["successful_steps"] / max(metadata["total_steps"], 1),
        "reliability": metadata["verification_success_rate"],
        "speed": metadata["total_steps"] / max(metadata["duration"], 1),
        "quality": metadata["average_confidence"]
    }
```

### **3. Error Analysis**

```python
def analyze_errors(run_data):
    errors = run_data.get("errors", [])
    
    error_types = {}
    for error in errors:
        error_msg = error["error"]
        if "not found" in error_msg.lower():
            error_types["element_not_found"] = error_types.get("element_not_found", 0) + 1
        elif "timeout" in error_msg.lower():
            error_types["timeout"] = error_types.get("timeout", 0) + 1
    
    return error_types
```

## 🚀 **Best Practices**

### **1. Always Use Context Managers**

```python
# ✅ Good: Automatic cleanup
with run_logging_session(user_goal) as run_logger:
    # Your code here
    pass

# ❌ Bad: Manual management (easy to forget cleanup)
run_logger = start_run_logging(user_goal)
try:
    # Your code here
    pass
finally:
    stop_run_logging()  # Easy to forget!
```

### **2. Use Descriptive Event Types**

```python
# ✅ Good: Clear, descriptive event types
run_logger.log_event("user_goal_received", "USER", {"goal": user_goal})
run_logger.log_event("episode_planning_started", "LLM-PLANNER", {"episode_id": episode_id})
run_logger.log_event("verification_completed", "LLM-VERIFIER", {"result": result})

# ❌ Bad: Vague event types
run_logger.log_event("event", "AGENT", {"data": "something happened"})
```

### **3. Include Rich Context in Errors**

```python
# ✅ Good: Rich error context
run_logger.log_error(
    "LLM-EXECUTOR",
    "Element not found",
    episode_id="episode_001",
    step_id="step_002",
    context={
        "expected_element": "search_button",
        "current_ui_state": ui_xml,
        "available_elements": ["menu_button", "back_button"],
        "retry_count": 3
    }
)

# ❌ Bad: Minimal error context
run_logger.log_error("LLM-EXECUTOR", "Element not found")
```

### **4. Log at Appropriate Levels**

```python
# ✅ Good: Appropriate severity levels
run_logger.log_event("step_completed", "AGENT", {"step": "launch_app"}, severity="INFO")
run_logger.log_error("AGENT", "Temporary failure", severity="WARNING")
run_logger.log_critical_failure("episode_001", step, failure_result)  # CRITICAL

# ❌ Bad: Overusing critical level
run_logger.log_event("step_completed", "AGENT", {"step": "launch_app"}, severity="CRITICAL")
```

## 🔧 **Configuration**

### **1. Environment Variables**

```bash
# .env file
LOG_LEVEL=INFO
MEMORY_STORE_PATH=memory_store
SCREENSHOT_DIR=logs/screenshots
```

### **2. Custom Log Directories**

```python
# Custom log directory
with run_logging_session("Test goal", logs_dir="custom_logs") as run_logger:
    # Logs saved to custom_logs/ directory
    pass
```

### **3. Custom Run IDs**

```python
# Custom run ID
with run_logging_session("Test goal", run_id="test_run_001") as run_logger:
    # Logs saved with custom run ID
    pass
```

## 📝 **Example Complete Integration**

```python
#!/usr/bin/env python3
"""
Complete example of integrating run logging with automation
"""

from core.run_logger_integration import run_logging_session
from core.logging_config import get_logger

log = get_logger("AUTOMATION-RUNNER")

def run_weather_app_automation():
    """Run complete weather app automation with full logging."""
    
    user_goal = "Launch weather app, search for New York, and check current conditions"
    
    with run_logging_session(user_goal) as run_logger:
        try:
            # Log automation start
            run_logger.log_event("automation_start", "RUNNER", {
                "goal": user_goal,
                "expected_steps": ["launch_app", "search", "verify_results"]
            })
            
            # Episode 1: Launch app
            episode_1 = "episode_001"
            run_logger.log_episode_start(episode_1, "Launch weather app")
            
            # Step 1: Launch app
            step_1 = {"action": "launch_app", "package": "com.weather.app", "step_id": "step_001"}
            result_1 = {"success": True, "duration": 2.1}
            ui_1 = "<UI>Weather app launched</UI>"
            
            run_logger.log_step_execution(episode_1, step_1, result_1, ui_1)
            
            # Verify app launch
            verification_1 = {
                "verified": True,
                "confidence": 0.9,
                "reason": "Weather app visible on screen"
            }
            run_logger.log_verification_report(episode_1, step_1, verification_1, ui_1)
            
            run_logger.log_episode_end(episode_1, "completed", "Weather app launched successfully")
            
            # Episode 2: Search for New York
            episode_2 = "episode_002"
            run_logger.log_episode_start(episode_2, "Search for New York weather")
            
            # Step 2: Tap search button
            step_2 = {"action": "tap", "resource_id": "search_button", "step_id": "step_002"}
            result_2 = {"success": True, "duration": 0.8}
            ui_2 = "<UI>Search field focused</UI>"
            
            run_logger.log_step_execution(episode_2, step_2, result_2, ui_2)
            
            # Step 3: Type "New York"
            step_3 = {"action": "type", "text": "New York", "step_id": "step_003"}
            result_3 = {"success": True, "duration": 1.2}
            ui_3 = "<UI>New York typed in search field</UI>"
            
            run_logger.log_step_execution(episode_2, step_3, result_3, ui_3)
            
            # Verify search results
            verification_2 = {
                "verified": True,
                "confidence": 0.85,
                "reason": "New York weather results displayed"
            }
            run_logger.log_verification_report(episode_2, step_3, verification_2, ui_3)
            
            run_logger.log_episode_end(episode_2, "completed", "New York weather found")
            
            # Log automation success
            run_logger.log_event("automation_complete", "RUNNER", {
                "status": "success",
                "total_episodes": 2,
                "total_steps": 3,
                "goal_achieved": True
            })
            
            return {"status": "success", "episodes": 2, "steps": 3}
            
        except Exception as e:
            # Log automation failure
            run_logger.log_error("RUNNER", f"Automation failed: {str(e)}", context={
                "error_type": "automation_failure",
                "current_episode": episode_2 if 'episode_2' in locals() else episode_1
            })
            
            # End current episode if active
            if 'episode_2' in locals() and episode_2 in run_logger.episodes:
                run_logger.log_episode_end(episode_2, "failed", f"Automation error: {str(e)}")
            
            raise

if __name__ == "__main__":
    try:
        result = run_weather_app_automation()
        print(f"✅ Automation completed: {result}")
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        print("Check the run log for detailed error information")
```

## 🎯 **Summary**

The **Run Logger System** provides:

1. **Complete Automation Audit Trail** - Every action, verification, and error is captured
2. **Single JSON Output** - All run data saved in one comprehensive, timestamped file
3. **Easy Integration** - Minimal code changes required for existing automation
4. **Rich Context** - Detailed information for debugging and analysis
5. **Performance Metrics** - Success rates, confidence scores, and timing data
6. **Screenshot Tracking** - Links to all verification screenshots
7. **Error Analysis** - Comprehensive error logging with context
8. **System Information** - Platform, environment, and configuration details

This creates a **production-ready logging system** that provides complete visibility into automation runs and enables comprehensive debugging and analysis! 🚀

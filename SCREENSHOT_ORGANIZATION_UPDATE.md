# 📸 **Screenshot Organization Update**

## 🎯 **Overview**

The screenshot organization has been updated to save screenshots in **run-specific folders** instead of a generalized screenshots folder. This provides better organization and makes it easier to associate screenshots with specific automation runs.

## 🔄 **Before vs After**

### **❌ Previous Organization (Generalized)**
```
logs/
├── screenshots/
│   ├── verification_1234567890.png
│   ├── step_1234567891.png
│   ├── verification_1234567892.png
│   └── ...
└── run_abc123/
    └── run_abc123_20250817_123456.json
```

**Problems:**
- All screenshots mixed together
- Hard to associate screenshots with specific runs
- Difficult to clean up old screenshots
- No clear organization

### **✅ New Organization (Run-Specific)**
```
logs/
├── run_abc123/
│   ├── screenshots/
│   │   ├── verification_1234567890.png
│   │   ├── step_1234567891.png
│   │   └── verification_1234567892.png
│   └── run_abc123_20250817_123456.json
├── run_def456/
│   ├── screenshots/
│   │   ├── verification_1234567893.png
│   │   └── step_1234567894.png
│   └── run_def456_20250817_124500.json
└── run_ghi789/
    ├── screenshots/
    │   └── verification_1234567895.png
    └── run_ghi789_20250817_125600.json
```

**Benefits:**
- Screenshots organized by run
- Easy to associate screenshots with specific automation sessions
- Simple cleanup (delete entire run folder)
- Clear organization structure

## 🔧 **Implementation Details**

### **1. Updated AndroidDevice.screenshot() Method**

The `screenshot()` method in `env/android_interface.py` now automatically detects the current run and saves to the appropriate directory:

```python
def screenshot(self, label: str) -> str:
    ts = int(time.time())
    
    # Try to get the current run logger to determine the run-specific directory
    try:
        from core.run_logger import get_run_logger
        run_logger = get_run_logger()
        if run_logger:
            # Use run-specific screenshot directory
            run_screenshot_dir = Path("logs") / f"run_{run_logger.run_id}" / "screenshots"
            run_screenshot_dir.mkdir(parents=True, exist_ok=True)
            local = run_screenshot_dir / f"{label}_{ts}.png"
            log.info(f"[ANDROID] Using run-specific screenshot directory: {run_screenshot_dir}")
        else:
            # Fallback to general screenshots directory
            local = SCREENSHOT_DIR / f"{label}_{ts}.png"
            log.info(f"[ANDROID] Using general screenshot directory: {SCREENSHOT_DIR}")
    except Exception as e:
        # Fallback to general screenshots directory if anything goes wrong
        local = SCREENSHOT_DIR / f"{label}_{ts}.png"
        log.warning(f"[ANDROID] Error getting run logger, using general directory: {e}")
    
    # ... rest of screenshot logic
```

### **2. Enhanced RunLogger Class**

The `RunLogger` class now automatically creates the screenshots directory structure:

```python
def __init__(self, run_id: str, user_goal: str, logs_dir: str = "logs"):
    self.run_id = run_id
    self.user_goal = user_goal
    self.logs_dir = Path(logs_dir)
    self.run_dir = self.logs_dir / f"run_{run_id}"
    self.screenshots_dir = self.run_dir / "screenshots"
    
    # Create directories
    self.run_dir.mkdir(parents=True, exist_ok=True)
    self.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # ... rest of initialization
```

### **3. New Helper Functions**

Added helper functions for easy access to screenshot directories:

```python
# In core/run_logger_integration.py
def get_screenshots_dir() -> Optional[Path]:
    """Get the current run's screenshots directory."""
    run_logger = get_run_logger()
    if run_logger:
        return run_logger.get_screenshots_dir()
    return None
```

## 🚀 **Usage Examples**

### **Automatic Screenshot Organization**

When you run the automation:

```bash
python -m runners.run_example --goal "Enable Wi-Fi in Android settings" --serial emulator-5554
```

Screenshots are automatically saved to:
```
logs/run_[RUN_ID]/screenshots/
```

### **Manual Screenshot Directory Access**

```python
from core.run_logger_integration import get_screenshots_dir

# Get current run's screenshot directory
screenshots_dir = get_screenshots_dir()
if screenshots_dir:
    print(f"Screenshots saved to: {screenshots_dir}")
    # List all screenshots in current run
    for screenshot in screenshots_dir.glob("*.png"):
        print(f"  - {screenshot.name}")
```

### **Verifier Agent Screenshots**

The verifier agent automatically saves verification screenshots to the run-specific directory:

```python
# In llm_verifier_agent.py
screenshot_path = self.device.screenshot(f"verification_step_{step_id}")
# Screenshot automatically saved to: logs/run_[RUN_ID]/screenshots/
```

## 📊 **Directory Structure Example**

After running an automation, you'll see:

```
logs/
└── run_7e0a6972/
    ├── screenshots/
    │   ├── verification_step_1_1755410748.png
    │   ├── verification_step_2_1755410750.png
    │   ├── step_execution_1755410752.png
    │   └── final_state_1755410755.png
    └── run_7e0a6972_20250817_011024.json
```

## 🔍 **Benefits**

### **1. Better Organization**
- Screenshots grouped by automation run
- Clear association between screenshots and automation sessions
- Logical folder structure

### **2. Easier Management**
- Delete entire run folder to clean up
- Copy specific run data for analysis
- Archive runs with all associated files

### **3. Improved Debugging**
- All run artifacts in one place
- Easy to correlate screenshots with log entries
- Better troubleshooting workflow

### **4. Professional Structure**
- Clean, organized file structure
- Easy to share complete run data
- Professional automation system appearance

## 🎉 **Result**

Now when you run automation:

1. **📁 Run folder created** - `logs/run_[RUN_ID]/`
2. **📸 Screenshots folder created** - `logs/run_[RUN_ID]/screenshots/`
3. **📊 All screenshots saved** to the run-specific folder
4. **🔄 Complete run data** organized in one place
5. **🧹 Easy cleanup** - delete entire run folder when done

**Screenshots are now perfectly organized by run, making your automation system much more professional and manageable!** 🚀

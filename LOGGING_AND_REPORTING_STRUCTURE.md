# 📁 **Logging and Reporting Structure: Complete Guide**

## 🎯 **Overview**

The multi-agent automation system generates comprehensive logs and reports that are stored in multiple locations for different purposes. This document explains where each type of data is saved and how to access it.

## 📂 **Directory Structure**

```
multi_agent_qa/
├── logs/                          # Main logging directory
│   ├── screenshots/              # Screenshots captured during verification
│   ├── aitw_enhanced_evaluation_report.json
│   ├── aitw_evaluation_report.json
│   └── sample_aitw_report.json
├── memory_store/                  # Persistent memory storage
│   └── narrative.pkl             # Long-term narrative memory
├── individual_results/            # Individual episode results
│   ├── demo1_result.json
│   ├── demo2_result.json
│   ├── demo3_result.json
│   ├── demo1_pipeline_result.json
│   └── demo2_pipeline_result.json
├── test_traces/                   # Test execution traces
├── __pycache__/                   # Python cache files
└── core/
    ├── logging_config.py          # Logging configuration
    └── config.py                  # Configuration settings
```

## 🔍 **Configuration Settings**

### **Environment Variables**
```bash
# .env file configuration
OPENAI_API_KEY=your_api_key_here
ANDROID_EMULATOR_SERIAL=emulator-5554
LOG_LEVEL=INFO
MEMORY_STORE_PATH=memory_store
SCREENSHOT_DIR=logs/screenshots
```

### **Default Configuration**
```yaml
# configs/default.yaml
agent:
  model: "gemini-2.5-flash-exp"
  temperature: 0.2
android:
  emulator_serial: null
log:
  level: "INFO"
```

## 📝 **Logging System**

### **1. Console Logging (Primary)**
```python
# core/logging_config.py
LOG_FORMAT = "%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s"

logging.basicConfig(
    level=log_level, 
    format=LOG_FORMAT, 
    handlers=[logging.StreamHandler(sys.stdout)]  # Console output only
)
```

**Location**: **Console/Standard Output** (not saved to files by default)
**Format**: `2024-01-15 10:30:45 | LLM-VERIFIER   | INFO     | Verification report published: True (confidence: 0.85)`

### **2. Agent-Specific Loggers**
```python
# Each agent has its own logger
log = get_logger("LLM-PLANNER")      # Planner agent logs
log = get_logger("LLM-EXECUTOR")     # Executor agent logs  
log = get_logger("LLM-VERIFIER")     # Verifier agent logs
log = get_logger("LLM-SUPERVISOR")   # Supervisor agent logs
```

## 📊 **Verification Reports**

### **1. Message Bus Reports**
```python
# Published to message bus (in-memory, not persisted)
publish(Message("LLM-VERIFIER", "verify-report", {
    "episode_id": episode_id,
    "step_id": step.get("step_id", "unknown"),
    "action": step.get("action", "unknown"),
    "verified": result.get("verified", False),
    "reason": result.get("reason", "No reason provided"),
    "confidence": result.get("confidence", 0.0),
    "ui_xml": ui_xml,
    "screenshot_path": screenshot_path,
    "analysis": result.get("analysis", {}),
    "timestamp": time.time()
}))
```

**Location**: **In-Memory Message Bus** (temporary, lost on restart)
**Purpose**: Inter-agent communication during execution

### **2. Verification Completion Messages**
```python
publish(Message("LLM-VERIFIER", "verification-complete", {
    "episode_id": episode_id,
    "verification_result": result,
    "ui_xml": ui_xml,
    "step": step,
    "timestamp": time.time()
}))
```

**Location**: **In-Memory Message Bus** (temporary, lost on restart)
**Purpose**: Trigger next action planning

### **3. Critical Failure Notifications**
```python
publish(Message("LLM-VERIFIER", "critical-failure", {
    "episode_id": episode_id,
    "step": step,
    "failure_result": failure_result,
    "timestamp": time.time(),
    "severity": "critical",
    "action_required": "manual_intervention"
}))
```

**Location**: **In-Memory Message Bus** (temporary, lost on restart)
**Purpose**: Alert system of unrecoverable failures

## 🖼️ **Screenshots**

### **Screenshot Capture**
```python
def _capture_screenshot(self, episode_id: str, step_id: str) -> str:
    """Capture a screenshot for verification."""
    try:
        # Use the device's screenshot capability
        screenshot_path = self.device.screenshot(f"{episode_id}_{step_id}")
        log.debug(f"Screenshot captured: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        log.warning(f"Screenshot capture failed: {e}")
        return None
```

**Location**: **`logs/screenshots/`** directory
**Naming Convention**: `{episode_id}_{step_id}.png`
**Example**: `episode_123_step_456.png`

### **Screenshot Configuration**
```python
# core/config.py
SCREENSHOT_DIR: str = get_env_var("SCREENSHOT_DIR", "logs/screenshots")
```

## 💾 **Memory Storage**

### **1. Episodic Memory (Short-term)**
```python
# core/memory.py
class EpisodicMemory:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}  # In-memory only
    
    def store(self, key: str, value: Any, tags: List[str] | None = None) -> None:
        self._store[key] = {"value": value, "tags": tags or []}
```

**Location**: **In-Memory** (temporary, lost on restart)
**Purpose**: Store episode-specific data during execution

### **2. Narrative Memory (Long-term)**
```python
class NarrativeMemory:
    def __init__(self) -> None:
        self.file = MEM_DIR / "narrative.pkl"  # Persistent file
        if self.file.exists():
            with self.file.open("rb") as f:
                self._store: Dict[str, Any] = pickle.load(f)
        else:
            self._store = {}
    
    def store(self, key: str, value: Any, tags: List[str] | None = None) -> None:
        self._store[key] = {"value": value, "tags": tags or []}
        with self.file.open("wb") as f:  # Save to disk
            pickle.dump(self._store, f)
```

**Location**: **`memory_store/narrative.pkl`** (persistent file)
**Purpose**: Long-term storage across episodes
**Format**: Python pickle file

### **3. Memory Store Configuration**
```python
# core/config.py
MEMORY_STORE_PATH: str = get_env_var("MEMORY_STORE_PATH", "memory_store")
```

## 📋 **Episode Results**

### **1. Individual Episode Results**
```python
# Stored in individual_results/ directory
demo1_result.json          # Demo 1 execution results
demo2_result.json          # Demo 2 execution results  
demo3_result.json          # Demo 3 execution results
demo1_pipeline_result.json # Demo 1 pipeline results
demo2_pipeline_result.json # Demo 2 pipeline results
```

**Location**: **`individual_results/`** directory
**Format**: JSON files
**Purpose**: Store detailed results of individual episode executions

### **2. Pipeline Results**
```python
# Stored in root directory
full_pipeline_results.json     # Complete pipeline execution results
all_individual_results.json    # Aggregated individual results
image_analysis_results.json    # Image analysis results
```

**Location**: **Project root directory**
**Format**: JSON files
**Purpose**: Store high-level pipeline and aggregated results

## 🔍 **Test Traces**

### **Test Execution Traces**
```python
# Stored in test_traces/ directory
# Contains detailed traces of test executions
```

**Location**: **`test_traces/`** directory
**Purpose**: Debug test executions and verify system behavior

## 📊 **AITW Evaluation Reports**

### **AITW-Specific Reports**
```python
# Stored in logs/ directory
aitw_enhanced_evaluation_report.json  # Enhanced evaluation results
aitw_evaluation_report.json           # Standard evaluation results
sample_aitw_report.json               # Sample evaluation report
```

**Location**: **`logs/`** directory
**Format**: JSON files
**Purpose**: Store AITW dataset evaluation results

## 🚨 **Current Limitations**

### **1. Log Persistence**
- **Console logs are NOT saved to files** by default
- Only in-memory message bus communication
- No persistent log files for debugging

### **2. Verification Report Persistence**
- Verification reports are only published to message bus
- No automatic saving to files
- Reports lost on system restart

### **3. Memory Persistence**
- Episodic memory is in-memory only
- Only narrative memory persists to disk
- Episode history lost on restart

## 🚀 **Recommended Improvements**

### **1. Add File Logging**
```python
# core/logging_config.py - Enhanced version
import logging
from pathlib import Path

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# File handler for persistent logs
file_handler = logging.FileHandler(log_dir / "system.log")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Root logger with both handlers
logging.basicConfig(
    level=log_level,
    handlers=[file_handler, console_handler]
)
```

### **2. Persistent Verification Reports**
```python
# agents/llm_verifier_agent.py - Enhanced version
def _publish_verification_report(self, episode_id: str, step: Dict[str, Any], result: Dict[str, Any], ui_xml: str, screenshot_path: Optional[str]):
    # ... existing code ...
    
    # Save to persistent file
    self._save_verification_report_to_file(episode_id, report)

def _save_verification_report_to_file(self, episode_id: str, report: Dict[str, Any]):
    """Save verification report to persistent file."""
    try:
        report_dir = Path("logs/verification_reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"{episode_id}_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info(f"Verification report saved to: {report_file}")
    except Exception as e:
        log.error(f"Failed to save verification report: {e}")
```

### **3. Enhanced Memory Persistence**
```python
# core/memory.py - Enhanced version
class EpisodicMemory:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._persistent_file = Path("memory_store/episodic.pkl")
        self._load_persistent_store()
    
    def _load_persistent_store(self):
        """Load persistent episodic memory from disk."""
        if self._persistent_file.exists():
            try:
                with self._persistent_file.open("rb") as f:
                    self._store = pickle.load(f)
            except Exception as e:
                log.warning(f"Failed to load persistent episodic memory: {e}")
    
    def store(self, key: str, value: Any, tags: List[str] | None = None) -> None:
        """Store with persistence."""
        self._store[key] = {"value": value, "tags": tags or []}
        self._save_persistent_store()
    
    def _save_persistent_store(self):
        """Save episodic memory to disk."""
        try:
            self._persistent_file.parent.mkdir(exist_ok=True)
            with self._persistent_file.open("wb") as f:
                pickle.dump(self._store, f)
        except Exception as e:
            log.error(f"Failed to save persistent episodic memory: {e}")
```

## 📁 **Complete Enhanced Directory Structure**

```
multi_agent_qa/
├── logs/                          # Enhanced logging directory
│   ├── system.log                 # Persistent system logs
│   ├── screenshots/               # Verification screenshots
│   ├── verification_reports/      # Persistent verification reports
│   │   ├── episode_123_1705311045.json
│   │   └── episode_124_1705311100.json
│   ├── aitw_enhanced_evaluation_report.json
│   ├── aitw_evaluation_report.json
│   └── sample_aitw_report.json
├── memory_store/                  # Enhanced memory storage
│   ├── narrative.pkl             # Long-term narrative memory
│   └── episodic.pkl              # Persistent episodic memory
├── individual_results/            # Individual episode results
├── test_traces/                   # Test execution traces
└── core/
    ├── logging_config.py          # Enhanced logging configuration
    └── config.py                  # Configuration settings
```

## 🎯 **Summary**

### **Current Storage Locations**
1. **Console Logs**: Standard output (not persisted)
2. **Screenshots**: `logs/screenshots/` directory
3. **Memory**: `memory_store/narrative.pkl` (narrative only)
4. **Results**: `individual_results/` and root directory JSON files
5. **Reports**: In-memory message bus (not persisted)

### **Recommended Improvements**
1. **Add file logging** to `logs/system.log`
2. **Persist verification reports** to `logs/verification_reports/`
3. **Persist episodic memory** to `memory_store/episodic.pkl`
4. **Structured logging** with different log levels per component

This would create a **fully persistent logging and reporting system** that survives system restarts and provides comprehensive debugging capabilities! 🚀

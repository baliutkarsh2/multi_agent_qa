# Android in the Wild (AITW) Integration

## Overview

The Android in the Wild (AITW) integration enhances the multi-agent QA system by incorporating real user interaction videos from the [Android in the Wild dataset](https://github.com/google-research/google-research/tree/master/android_in_the_wild). This integration provides a comprehensive evaluation framework that tests the system's ability to reproduce real-world user interactions and handle the complexity and diversity of actual Android applications.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AITW Integration Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Video     │    │   Task      │    │   Trace     │         │
│  │  Analyzer   │    │ Generator   │    │ Recorder    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             │                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   AITW      │    │   Multi-    │    │ Evaluation  │         │
│  │ Evaluator   │    │   Agent     │    │   Report    │         │
│  │             │    │   System    │    │ Generator   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Video Processing Pipeline                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Frame     │    │   UI State  │    │   Action    │         │
│  │ Extraction  │    │  Analysis   │    │ Extraction  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Evaluation Metrics                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Accuracy   │    │ Robustness  │    │Generalization│        │
│  │   Score     │    │   Score     │    │   Score     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Video Input**: AITW videos containing real user interactions
2. **Video Analysis**: Extract frames, UI states, and user actions
3. **Task Generation**: Generate natural language task prompts
4. **Multi-Agent Execution**: Run the multi-agent system to reproduce the flow
5. **Trace Recording**: Capture detailed execution traces
6. **Evaluation**: Compare agent performance with ground truth
7. **Reporting**: Generate comprehensive evaluation reports

## Components

### 1. Video Analyzer (`evaluation/aitw_evaluator.py`)

The `AITWVideoAnalyzer` processes video files to extract:

- **Frame Analysis**: Extracts frames with timestamps using OpenCV
- **UI State Detection**: Analyzes UI elements, buttons, text regions
- **Action Extraction**: Identifies user actions through frame differences
- **Task Completion Detection**: Determines if the original task was completed

```python
from evaluation.aitw_evaluator import AITWVideoAnalyzer

analyzer = AITWVideoAnalyzer()
video_trace = analyzer.analyze_video(Path("user_interaction.mp4"))
```

### 2. Task Prompt Generator (`evaluation/aitw_evaluator.py`)

The `TaskPromptGenerator` uses LLM reasoning to convert video analysis into natural language task descriptions:

```python
from evaluation.aitw_evaluator import TaskPromptGenerator

generator = TaskPromptGenerator()
task_prompt = generator.generate_task_prompt(video_trace)
# Output: "Enable Wi-Fi in Android settings by navigating to Settings > Wi-Fi and toggling the switch"
```

### 3. Trace Recorder (`evaluation/trace_recorder.py`)

The `TraceRecorder` captures detailed execution traces from the multi-agent system:

- **Action Tracking**: Records each action with timestamps and UI states
- **Success Monitoring**: Tracks success/failure of each action
- **Episode Management**: Manages complete episode lifecycle
- **Trace Analysis**: Provides analysis and comparison tools

```python
from evaluation.trace_recorder import TraceRecorder, TraceAnalyzer

recorder = TraceRecorder()
# Automatically records traces during execution

# Analyze traces
analyzer = TraceAnalyzer()
analysis = analyzer.analyze_trace(episode_trace)
```

### 4. AITW Evaluator (`evaluation/aitw_evaluator.py`)

The `AITWEvaluator` provides comprehensive evaluation metrics:

- **Accuracy Score**: How well the agent reproduced the video flow
- **Robustness Score**: How well the agent handled variations
- **Generalization Score**: How well the agent generalizes to new scenarios
- **Action Similarity**: Similarity between agent and human actions
- **UI State Similarity**: Similarity between reached UI states

```python
from evaluation.aitw_evaluator import AITWEvaluator

evaluator = AITWEvaluator()
score = evaluator.evaluate_video(video_path, agent_trace)
```

### 5. AITW Runner (`runners/aitw_runner.py`)

The `AITWRunner` orchestrates the complete evaluation pipeline:

```python
from runners.aitw_runner import AITWRunner

runner = AITWRunner(video_dir="aitw_videos", num_videos=3)
results = runner.run_evaluation()
```

## Usage

### Basic Usage

1. **Setup Sample Videos**:
```bash
python -m runners.aitw_runner --setup-only
```

2. **Run Evaluation**:
```bash
python -m runners.aitw_runner --num-videos 3
```

3. **Custom Video Directory**:
```bash
python -m runners.aitw_runner --video-dir your_videos --num-videos 5
```

### Advanced Usage

#### Custom Video Analysis

```python
from evaluation.aitw_evaluator import AITWVideoAnalyzer, VideoTrace
from pathlib import Path

# Analyze a specific video
analyzer = AITWVideoAnalyzer()
video_trace = analyzer.analyze_video(Path("custom_video.mp4"))

# Access analysis results
print(f"Video duration: {video_trace.timestamps[-1]:.2f}s")
print(f"User actions: {len(video_trace.user_actions)}")
print(f"Task completed: {video_trace.task_completion}")
```

#### Custom Task Generation

```python
from evaluation.aitw_evaluator import TaskPromptGenerator

generator = TaskPromptGenerator()
task_prompt = generator.generate_task_prompt(video_trace)
print(f"Generated task: {task_prompt}")
```

#### Trace Analysis

```python
from evaluation.trace_recorder import TraceRecorder, TraceAnalyzer

# Load existing trace
recorder = TraceRecorder()
episode_trace = recorder.load_trace("episode_123")

# Analyze trace
analyzer = TraceAnalyzer()
analysis = analyzer.analyze_trace(episode_trace)

print(f"Success rate: {analysis['success_rate']:.2f}")
print(f"Total actions: {analysis['total_actions']}")
print(f"Task completed: {analysis['task_completed']}")
```

#### Custom Evaluation

```python
from evaluation.aitw_evaluator import AITWEvaluator, AgentTrace

# Create custom agent trace
agent_trace = AgentTrace(
    episode_id="custom_episode",
    actions=[{"action": "tap", "success": True}],
    ui_states=[{"xml": "<ui_state>"}],
    timestamps=[0.0, 2.0],
    task_completion=True,
    success_rate=1.0,
    duration=5.0
)

# Evaluate against video
evaluator = AITWEvaluator()
score = evaluator.evaluate_video(Path("video.mp4"), agent_trace)

print(f"Accuracy: {score.accuracy_score:.3f}")
print(f"Robustness: {score.robustness_score:.3f}")
print(f"Generalization: {score.generalization_score:.3f}")
```

## Evaluation Metrics

### Accuracy Score
Measures how accurately the agent reproduced the video flow:
- **Calculation**: Overlap between video and agent action types
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How well the agent followed the same action sequence

### Robustness Score
Measures how well the agent handled variations and errors:
- **Calculation**: 1.0 - (error_count / total_actions)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How resilient the agent was to failures

### Generalization Score
Measures how well the agent adapts to different scenarios:
- **Calculation**: Efficiency ratio between video and agent execution
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How well the agent generalizes to new situations

### Action Similarity
Measures similarity between agent and human actions:
- **Calculation**: Sequence alignment of action types
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How similar the action sequences are

### UI State Similarity
Measures similarity between reached UI states:
- **Calculation**: Jaccard similarity of UI state properties
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: How similar the final UI states are

## Sample Videos

The system includes sample video metadata for common Android tasks:

1. **Settings Wi-Fi Enable** (`settings_wifi_enable.mp4`)
   - Duration: 8.5s
   - Expected actions: tap, tap, verify
   - Description: User enabling Wi-Fi in Android settings

2. **Chrome Search Weather** (`chrome_search_weather.mp4`)
   - Duration: 12.3s
   - Expected actions: tap, type, press_key, verify
   - Description: User searching for weather in Chrome browser

3. **Gmail Open Inbox** (`gmail_open_inbox.mp4`)
   - Duration: 6.7s
   - Expected actions: launch_app, tap, verify
   - Description: User opening Gmail and accessing inbox

4. **Calculator Basic Math** (`calculator_basic_math.mp4`)
   - Duration: 15.2s
   - Expected actions: launch_app, tap, tap, tap, verify
   - Description: User performing basic math operations

5. **Camera Take Photo** (`camera_take_photo.mp4`)
   - Duration: 9.8s
   - Expected actions: launch_app, tap, wait, verify
   - Description: User taking a photo with the camera app

## Output Reports

### Evaluation Report Structure

```json
{
  "evaluation_summary": {
    "total_videos": 3,
    "video_names": ["video1.mp4", "video2.mp4", "video3.mp4"],
    "evaluation_timestamp": 1640995200.0,
    "system_version": "multi_agent_qa_v1.0"
  },
  "aggregate_scores": {
    "average_accuracy": 0.85,
    "average_robustness": 0.92,
    "average_generalization": 0.78,
    "task_completion_rate": 0.93,
    "average_duration": 12.5
  },
  "individual_results": [
    {
      "video_name": "video1.mp4",
      "scores": {
        "accuracy_score": 0.88,
        "robustness_score": 0.95,
        "generalization_score": 0.82,
        "task_completion_rate": 1.0,
        "average_duration": 10.2,
        "action_similarity": 0.90,
        "ui_state_similarity": 0.85
      }
    }
  ]
}
```

### Trace Files

Execution traces are saved as JSON files in `logs/traces/`:

```json
{
  "episode_id": "episode_123",
  "user_goal": "Enable Wi-Fi in Android settings",
  "start_time": 1640995200.0,
  "end_time": 1640995205.0,
  "actions": [
    {
      "timestamp": 1640995201.0,
      "action_type": "launch_app",
      "action_data": {"package": "com.android.settings"},
      "ui_state_before": "<home_screen>",
      "ui_state_after": "<settings_main>",
      "success": true,
      "error": null,
      "duration": 1.0
    }
  ],
  "final_ui_state": "<wifi_enabled>",
  "task_completed": true,
  "completion_reason": "Wi-Fi successfully enabled",
  "metadata": {
    "recorder_version": "1.0",
    "recording_timestamp": 1640995205.0,
    "total_actions": 3,
    "episode_duration": 5.0
  }
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_aitw_integration.py
```

This will test:
- Video analysis components
- Task generation functionality
- Trace recording and analysis
- Evaluation metrics calculation
- Complete pipeline integration

## Integration with Multi-Agent System

The AITW integration seamlessly works with the existing multi-agent architecture:

1. **Planner Agent**: Receives generated task prompts from video analysis
2. **Executor Agent**: Performs actions while being monitored by trace recorder
3. **Verifier Agent**: Validates actions against expected outcomes
4. **Supervisor Agent**: Coordinates the evaluation process

### Message Flow

```
Video Analysis → Task Generation → Planner Agent → Executor Agent
                                                      ↓
Trace Recorder ← Execution Reports ← Verifier Agent ← Supervisor Agent
      ↓
Evaluation Report
```

## Benefits

### Enhanced Evaluation
- **Real-world Testing**: Uses actual user interaction patterns
- **Diverse Scenarios**: Covers various apps and interaction types
- **Robustness Testing**: Handles real-world complexity and variations

### Improved Training
- **Ground Truth Data**: Provides real user behavior as reference
- **Performance Benchmarking**: Establishes baseline performance metrics
- **Iterative Improvement**: Enables continuous system enhancement

### Research Applications
- **Human-AI Comparison**: Compare agent behavior with human behavior
- **Generalization Studies**: Test system performance across different scenarios
- **Robustness Analysis**: Evaluate system resilience to variations

## Future Enhancements

1. **Advanced Video Analysis**: Integration with OCR and computer vision
2. **Real-time Evaluation**: Live evaluation during agent execution
3. **Cross-platform Support**: Extension to iOS and web applications
4. **Automated Dataset Management**: Tools for managing and curating video datasets
5. **Performance Optimization**: Faster video processing and analysis

## References

- [Android in the Wild Dataset](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
- [Multi-Agent QA System Architecture](README.md)
- [Evaluation Framework](evaluation/)
- [Trace Recording System](evaluation/trace_recorder.py)

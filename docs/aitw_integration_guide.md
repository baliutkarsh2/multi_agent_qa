# 🚀 Android in the Wild (AITW) Integration Guide

This guide explains how to use the **Android in the Wild (AITW) dataset integration** with your multi-agent QA system. The AITW dataset contains real user interaction videos from Android devices, making it perfect for evaluating how well your AI agents can reproduce human behavior.

## 📋 Table of Contents

- [🎯 What is AITW?](#-what-is-aitw)
- [🔧 Setup & Installation](#-setup--installation)
- [💻 Usage Examples](#-usage-examples)
- [📊 Understanding the Data](#-understanding-the-data)
- [🎨 Visualization Tools](#-visualization-tools)
- [🧪 Evaluation & Testing](#-evaluation--testing)
- [🔍 Troubleshooting](#-troubleshooting)
- [📚 Advanced Usage](#-advanced-usage)

## 🎯 What is AITW?

**Android in the Wild (AITW)** is a dataset created by Google Research that contains:

- **Real user interaction videos** from Android devices
- **UI state snapshots** at each interaction step
- **Action annotations** (tap, type, scroll, etc.)
- **Multiple app categories** (general, Google apps, shopping, etc.)
- **Diverse user behaviors** and interaction patterns

This dataset is perfect for:
- **Training AI agents** to understand real user behavior
- **Evaluating automation systems** against ground truth
- **Research in mobile UI automation** and testing
- **Benchmarking multi-agent systems** for Android interaction

## 🔧 Setup & Installation

### Prerequisites

- **Python 3.7+** (required for TensorFlow and modern features)
- **Git** (for cloning the Google Research repository)
- **pip** (for installing dependencies)
- **Internet connection** (for accessing the dataset)

### Quick Setup

Run the automated setup script:

```bash
python setup_aitw_dataset.py
```

This script will:
1. ✅ Check Python version and dependencies
2. 📦 Install TensorFlow automatically
3. 📥 Clone the Google Research repository
4. 🧪 Test all imports and dataset access
5. 📝 Create a demo script for testing

### Manual Setup

If you prefer manual setup:

```bash
# 1. Install TensorFlow
pip install tensorflow

# 2. Clone Google Research repository
git clone https://github.com/google-research/google-research.git

# 3. Test the setup
python -c "
import tensorflow as tf
import sys
sys.path.append('./google-research')
from android_in_the_wild import visualization_utils
print('✅ Setup successful!')
"
```

## 💻 Usage Examples

### Basic Data Loading

```python
from evaluation.aitw_data_loader import AITWDataLoader

# Initialize loader for Google Apps dataset
loader = AITWDataLoader("google_apps")

# Get sample episodes
episodes = loader.get_sample_episodes(3)

for episode in episodes:
    print(f"Episode: {episode['episode_id']}")
    print(f"App: {episode['app_package']}")
    print(f"Steps: {len(episode['steps'])}")
    print(f"Actions: {episode['metadata']['unique_actions']}")
```

### Episode Visualization

```python
# Visualize an episode with annotations
raw_dataset = loader.get_raw_dataset()
tf_episode = loader.get_episode(raw_dataset)

# Save visualization to file
loader.visualize_episode(tf_episode, "episode_visualization.png")

# Or display interactively
loader.visualize_episode(tf_episode)
```

### Running Evaluation

```bash
# Demo mode
python -m runners.aitw_enhanced_runner --demo

# Evaluate specific dataset
python -m runners.aitw_enhanced_runner --dataset google_apps --episodes 5

# Use local videos only
python -m runners.aitw_enhanced_runner --local-only
```

## 📊 Understanding the Data

### Dataset Structure

The AITW dataset is organized into categories:

| Category | Description | Use Case |
|----------|-------------|----------|
| `general` | General Android app interactions | Broad testing |
| `google_apps` | Google app interactions (Gmail, Chrome, etc.) | Google ecosystem testing |
| `install` | App installation flows | Installation testing |
| `single` | Single-app focused interactions | App-specific testing |
| `web_shopping` | E-commerce interactions | Shopping app testing |

### Episode Format

Each episode contains:

```python
{
    "episode_id": "unique_identifier",
    "app_package": "com.example.app",
    "steps": [
        {
            "step_id": 0,
            "timestamp": 0,
            "action": "launch",
            "ui_state": {
                "xml": "<UI hierarchy XML>",
                "screenshot": None  # Not available in TFRecord
            }
        },
        # ... more steps
    ],
    "metadata": {
        "actions": ["launch", "tap", "type", "verify"],
        "unique_actions": ["launch", "tap", "type", "verify"],
        "num_steps": 4
    }
}
```

### Action Types

Common actions in the dataset:

- **`launch`**: App launch
- **`tap`**: Touch interaction
- **`type`**: Text input
- **`scroll`**: Scrolling gesture
- **`press_key`**: Hardware key press
- **`verify`**: State verification

## 🎨 Visualization Tools

### Episode Visualization

The AITW visualization tools provide rich visual representations:

```python
from android_in_the_wild import visualization_utils

# Plot episode with annotations
visualization_utils.plot_episode(
    episode, 
    show_annotations=True,  # Show action annotations
    show_actions=True        # Show action labels
)
```

### Custom Visualizations

Create custom visualizations for your analysis:

```python
import matplotlib.pyplot as plt

# Create custom episode timeline
fig, ax = plt.subplots(figsize=(15, 8))
steps = episode['steps']
actions = [step['action'] for step in steps]

ax.bar(range(len(actions)), [1] * len(actions))
ax.set_xticks(range(len(actions)))
ax.set_xticklabels(actions, rotation=45)
ax.set_title(f"Episode {episode['episode_id']} - Action Timeline")
ax.set_ylabel("Actions")
plt.tight_layout()
plt.savefig("action_timeline.png")
```

## 🧪 Evaluation & Testing

### Multi-Agent Evaluation

Your system can now evaluate AI agents against real user behavior:

```python
from runners.aitw_enhanced_runner import AITWEnhancedRunner

# Initialize enhanced runner
runner = AITWEnhancedRunner(
    dataset_name="google_apps",
    num_episodes=5,
    use_official_dataset=True
)

# Run evaluation
results = runner.run_official_dataset_evaluation()

# Results include:
# - Accuracy scores
# - Robustness metrics
# - Generalization scores
# - Task completion rates
```

### Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy Score**: How well agents reproduce user actions
- **Robustness Score**: Error handling and recovery capability
- **Generalization Score**: Adaptation to different scenarios
- **Action Similarity**: Comparison with human behavior patterns
- **UI State Similarity**: Final state achievement

### Custom Evaluation

Create custom evaluation criteria:

```python
def custom_evaluator(episode, agent_trace):
    """Custom evaluation logic."""
    
    # Compare action sequences
    human_actions = [step['action'] for step in episode['steps']]
    agent_actions = [action['action'] for action in agent_trace.actions]
    
    # Calculate sequence similarity
    similarity = calculate_sequence_similarity(human_actions, agent_actions)
    
    return {
        'custom_score': similarity,
        'action_sequence_match': similarity > 0.8
    }
```

## 🔍 Troubleshooting

### Common Issues

#### 1. TensorFlow Import Error
```bash
# Solution: Install TensorFlow
pip install tensorflow

# Or for GPU support
pip install tensorflow-gpu
```

#### 2. Google Research Repository Not Found
```bash
# Solution: Clone the repository
git clone https://github.com/google-research/google-research.git
```

#### 3. Dataset Access Issues
```bash
# Check internet connection
# Verify Google Cloud Storage access
# Try different dataset categories
```

#### 4. Memory Issues
```python
# Reduce batch size
loader = AITWDataLoader("google_apps")
episodes = loader.get_sample_episodes(1)  # Start with 1 episode

# Use smaller datasets
loader = AITWDataLoader("single")  # Smaller category
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use the system's logging
from core.logging_config import get_logger
log = get_logger("AITW-DEBUG")
log.setLevel(logging.DEBUG)
```

## 📚 Advanced Usage

### Custom Data Processing

```python
class CustomAITWProcessor:
    def __init__(self, dataset_name):
        self.loader = AITWDataLoader(dataset_name)
    
    def filter_episodes(self, min_steps=5, max_steps=20):
        """Filter episodes by step count."""
        episodes = self.loader.get_sample_episodes(100)
        return [ep for ep in episodes 
                if min_steps <= len(ep['steps']) <= max_steps]
    
    def extract_patterns(self, episodes):
        """Extract common interaction patterns."""
        patterns = {}
        for episode in episodes:
            actions = tuple(episode['metadata']['actions'])
            patterns[actions] = patterns.get(actions, 0) + 1
        return patterns
```

### Integration with Other Tools

```python
# Export to different formats
import pandas as pd

def export_to_csv(episodes, filename):
    """Export episodes to CSV for analysis."""
    data = []
    for episode in episodes:
        for step in episode['steps']:
            data.append({
                'episode_id': episode['episode_id'],
                'step_id': step['step_id'],
                'action': step['action'],
                'app_package': episode['app_package']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df

# Use with pandas
episodes = loader.get_sample_episodes(10)
df = export_to_csv(episodes, "aitw_episodes.csv")
print(df.groupby('action').count())
```

### Performance Optimization

```python
# Parallel processing for large datasets
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_episode_batch(episodes):
    """Process multiple episodes in parallel."""
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_single_episode, ep) for ep in episodes]
        results = [future.result() for future in futures]
    return results

# Batch processing
all_episodes = loader.get_sample_episodes(100)
batch_size = 10
batches = [all_episodes[i:i+batch_size] 
           for i in range(0, len(all_episodes), batch_size)]

for batch in batches:
    results = process_episode_batch(batch)
    # Process results...
```

## 🎉 Next Steps

Now that you have AITW integration:

1. **Run the demo** to verify everything works
2. **Explore different datasets** to find relevant episodes
3. **Evaluate your agents** against real user behavior
4. **Analyze results** to improve agent performance
5. **Contribute improvements** to the evaluation system

The AITW dataset provides a **gold standard** for evaluating Android automation systems. By comparing your AI agents against real user behavior, you can:

- **Identify weaknesses** in your automation logic
- **Improve action selection** based on human patterns
- **Validate system robustness** with diverse scenarios
- **Benchmark performance** against industry standards

Happy testing! 🚀📱

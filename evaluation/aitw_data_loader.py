"""
Android in the Wild (AITW) Data Loader
Provides access to the official Google Research AITW dataset and visualization tools.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import logging

# Try to import TensorFlow and AITW tools
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

try:
    # Add google-research to path if available
    google_research_path = Path("./google-research")
    if google_research_path.exists():
        sys.path.append(str(google_research_path))
        from android_in_the_wild import visualization_utils
        AITW_VISUALIZATION_AVAILABLE = True
    else:
        AITW_VISUALIZATION_AVAILABLE = False
        logging.warning("Google Research repository not found. Clone with: git clone https://github.com/google-research/google-research.git")
except ImportError:
    AITW_VISUALIZATION_AVAILABLE = False
    logging.warning("AITW visualization tools not available")

from core.logging_config import get_logger

log = get_logger("AITW-DATA-LOADER")

class AITWDataLoader:
    """Loader for Android in the Wild dataset from Google Research."""
    
    def __init__(self, dataset_name: str = "google_apps"):
        self.dataset_name = dataset_name
        self.dataset_directories = {
            'general': 'gs://gresearch/android-in-the-wild/general/*',
            'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
            'install': 'gs://gresearch/android-in-the-wild/install/*',
            'single': 'gs://gresearch/android-in-the-wild/single/*',
            'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
        }
        self.filenames = []  # Initialize filenames attribute
        self.dataset_path = None
        
        if not TENSORFLOW_AVAILABLE:
            log.error("TensorFlow not available. Cannot load AITW dataset.")
            return
            
        self._setup_dataset()
    
    def _setup_dataset(self):
        """Set up the dataset access."""
        try:
            # Try multiple access methods
            if self._try_hf_access():
                log.info("Successfully loaded AITW dataset via Hugging Face")
                return
            elif self._try_tfds_access():
                log.info("Successfully loaded AITW dataset via TFDS")
                return
            elif self._try_local_download():
                log.info("Successfully loaded AITW dataset via local download")
                return
            elif self._try_gs_access():
                log.info("Successfully loaded AITW dataset via Google Cloud Storage")
                return
            else:
                log.info("Falling back to demo mode with synthetic data")
                self._setup_demo_mode()
                
        except Exception as e:
            log.error(f"Error setting up dataset: {e}")
            self._setup_demo_mode()
    
    def _setup_demo_mode(self):
        """Set up demo mode with synthetic AITW-like data."""
        log.info("Setting up demo mode with synthetic AITW data")
        self.dataset_path = "demo"
        
        # Create synthetic episodes that mimic AITW structure
        self.demo_episodes = self._create_demo_episodes()
        log.info(f"Created {len(self.demo_episodes)} demo episodes")
    
    def _create_demo_episodes(self) -> List[Dict[str, Any]]:
        """Create synthetic episodes for demo purposes."""
        demo_episodes = []
        
        # Create different types of episodes based on dataset name
        if self.dataset_name == "google_apps":
            episode_types = [
                {
                    "episode_id": "demo_gmail_001",
                    "app_package": "com.google.android.gm",
                    "steps": [
                        {"action": "launch", "ui_element": "gmail_icon", "timestamp": 0},
                        {"action": "tap", "ui_element": "compose_button", "timestamp": 1},
                        {"action": "type", "ui_element": "to_field", "timestamp": 2, "text": "test@example.com"},
                        {"action": "type", "ui_element": "subject_field", "timestamp": 3, "text": "Test Email"},
                        {"action": "type", "ui_element": "body_field", "timestamp": 4, "text": "This is a test email."},
                        {"action": "tap", "ui_element": "send_button", "timestamp": 5}
                    ]
                },
                {
                    "episode_id": "demo_chrome_001", 
                    "app_package": "com.android.chrome",
                    "steps": [
                        {"action": "launch", "ui_element": "chrome_icon", "timestamp": 0},
                        {"action": "tap", "ui_element": "address_bar", "timestamp": 1},
                        {"action": "type", "ui_element": "address_bar", "timestamp": 2, "text": "google.com"},
                        {"action": "tap", "ui_element": "go_button", "timestamp": 3},
                        {"action": "tap", "ui_element": "search_box", "timestamp": 4},
                        {"action": "type", "ui_element": "search_box", "timestamp": 5, "text": "android automation"}
                    ]
                },
                {
                    "episode_id": "demo_maps_001",
                    "app_package": "com.google.android.apps.maps",
                    "steps": [
                        {"action": "launch", "ui_element": "maps_icon", "timestamp": 0},
                        {"action": "tap", "ui_element": "search_bar", "timestamp": 1},
                        {"action": "type", "ui_element": "search_bar", "timestamp": 2, "text": "Purdue University"},
                        {"action": "tap", "ui_element": "search_result", "timestamp": 3},
                        {"action": "tap", "ui_element": "directions_button", "timestamp": 4}
                    ]
                }
            ]
        else:
            # Generic episodes for other dataset types
            episode_types = [
                {
                    "episode_id": "demo_generic_001",
                    "app_package": "com.example.app",
                    "steps": [
                        {"action": "launch", "ui_element": "app_icon", "timestamp": 0},
                        {"action": "tap", "ui_element": "main_button", "timestamp": 1},
                        {"action": "swipe", "ui_element": "content_area", "timestamp": 2, "direction": "up"},
                        {"action": "tap", "ui_element": "menu_button", "timestamp": 3}
                    ]
                }
            ]
        
        for episode_data in episode_types:
            # Convert to the format expected by the system
            episode = {
                'episode_id': episode_data['episode_id'],
                'app_package': episode_data['app_package'],
                'steps': episode_data['steps'],
                'metadata': {
                    'episode_id': episode_data['episode_id'],
                    'app_package': episode_data['app_package'],
                    'num_steps': len(episode_data['steps']),
                    'dataset_name': self.dataset_name,
                    'actions': [step['action'] for step in episode_data['steps']],
                    'unique_actions': list(set([step['action'] for step in episode_data['steps']]))
                }
            }
            demo_episodes.append(episode)
        
        return demo_episodes
    
    def _try_hf_access(self) -> bool:
        """Try to access AITW dataset via Hugging Face datasets."""
        try:
            from datasets import load_dataset
            
            log.info("Attempting to load AITW dataset via Hugging Face...")
            
            # Try to load the dataset from Hugging Face
            # The AITW dataset is available as "google-research/android-in-the-wild"
            dataset = load_dataset("google-research/android-in-the-wild", split="train")
            
            if dataset and len(dataset) > 0:
                log.info(f"Successfully loaded {len(dataset)} episodes via Hugging Face")
                self.dataset_path = "hf"
                return True
            else:
                log.warning("Hugging Face returned empty dataset")
                return False
                
        except ImportError:
            log.warning("Hugging Face datasets not available. Install with: pip install datasets")
            return False
        except Exception as e:
            log.warning(f"Hugging Face access failed: {e}")
            return False
    
    def _try_tfds_access(self) -> bool:
        """Try to access AITW dataset via TensorFlow Datasets."""
        try:
            import tensorflow_datasets as tfds
            
            log.info("Attempting to load AITW dataset via TFDS...")
            
            # Try to load the dataset
            dataset = tfds.load('android_in_the_wild', split='train', as_supervised=False)
            
            # Convert to list for easier processing
            episodes = list(dataset.take(10))  # Take first 10 episodes as test
            
            if episodes:
                log.info(f"Successfully loaded {len(episodes)} episodes via TFDS")
                self.dataset_path = "tfds"
                return True
            else:
                log.warning("TFDS returned empty dataset")
                return False
                
        except ImportError:
            log.warning("TensorFlow Datasets not available. Install with: pip install tensorflow-datasets")
            return False
        except Exception as e:
            log.warning(f"TFDS access failed: {e}")
            return False
    
    def _try_local_download(self) -> bool:
        """Try to download AITW dataset locally."""
        try:
            import requests
            import tempfile
            import os
            
            log.info("Attempting to download AITW dataset locally...")
            
            # Create local dataset directory
            dataset_dir = Path("./aitw_dataset")
            dataset_dir.mkdir(exist_ok=True)
            
            # Check if we already have the dataset
            if self._check_local_dataset(dataset_dir):
                log.info("Local dataset already exists")
                self.dataset_path = str(dataset_dir)
                return True
            
            # Download dataset files
            base_url = "https://storage.googleapis.com/gresearch/android-in-the-wild"
            dataset_files = {
                'google_apps': f"{base_url}/google_apps/",
                'general': f"{base_url}/general/",
                'install': f"{base_url}/install/",
                'single': f"{base_url}/single/",
                'web_shopping': f"{base_url}/web_shopping/"
            }
            
            target_dir = dataset_dir / self.dataset_name
            target_dir.mkdir(exist_ok=True)
            
            # Try to download a sample file to test access
            sample_url = f"{dataset_files[self.dataset_name]}00000.tfrecord"
            
            try:
                response = requests.get(sample_url, timeout=30)
                if response.status_code == 200:
                    log.info("Successfully accessed AITW dataset via HTTP")
                    self.dataset_path = str(target_dir)
                    return True
                else:
                    log.warning(f"HTTP access failed with status {response.status_code}")
                    return False
            except Exception as e:
                log.warning(f"HTTP download failed: {e}")
                return False
                
        except ImportError:
            log.warning("Requests not available. Install with: pip install requests")
            return False
        except Exception as e:
            log.warning(f"Local download failed: {e}")
            return False
    
    def _try_gs_access(self) -> bool:
        """Try to access AITW dataset via Google Cloud Storage."""
        try:
            dataset_path = self.dataset_directories.get(self.dataset_name)
            if not dataset_path:
                log.error(f"Unknown dataset: {self.dataset_name}")
                return False
                
            log.info(f"Attempting to access AITW dataset via GCS: {self.dataset_name}")
            self.filenames = tf.io.gfile.glob(dataset_path)
            log.info(f"Found {len(self.filenames)} dataset files via GCS")
            
            if self.filenames:
                self.dataset_path = "gcs"
                return True
            else:
                return False
                
        except Exception as e:
            log.warning(f"GCS access failed: {e}")
            return False
    
    def _check_local_dataset(self, dataset_dir: Path) -> bool:
        """Check if local dataset exists and is valid."""
        try:
            target_dir = dataset_dir / self.dataset_name
            if not target_dir.exists():
                return False
            
            # Check for TFRecord files
            tfrecord_files = list(target_dir.glob("*.tfrecord"))
            if len(tfrecord_files) > 0:
                log.info(f"Found {len(tfrecord_files)} local TFRecord files")
                self.filenames = [str(f) for f in tfrecord_files]
                return True
            
            return False
            
        except Exception as e:
            log.warning(f"Local dataset check failed: {e}")
            return False
    
    def get_raw_dataset(self) -> Iterator:
        """Get the raw dataset iterator based on access method."""
        try:
            if self.dataset_path == "demo":
                return self._get_demo_dataset()
            elif self.dataset_path == "hf":
                return self._get_hf_dataset()
            elif self.dataset_path == "tfds":
                return self._get_tfds_dataset()
            elif self.dataset_path == "gcs" or (self.dataset_path and self.dataset_path.startswith("./")):
                return self._get_tfrecord_dataset()
            else:
                log.error("No valid dataset access method available")
                return iter([])
                
        except Exception as e:
            log.error(f"Error creating dataset iterator: {e}")
            return iter([])
    
    def _get_demo_dataset(self) -> Iterator:
        """Get dataset iterator from demo episodes."""
        try:
            return iter(self.demo_episodes)
        except Exception as e:
            log.error(f"Error creating demo dataset iterator: {e}")
            return iter([])
    
    def _get_hf_dataset(self) -> Iterator:
        """Get dataset iterator from Hugging Face datasets."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("google-research/android-in-the-wild", split="train")
            return iter(dataset)
        except Exception as e:
            log.error(f"Error loading Hugging Face dataset: {e}")
            return iter([])
    
    def _get_tfds_dataset(self) -> Iterator:
        """Get dataset iterator from TensorFlow Datasets."""
        try:
            import tensorflow_datasets as tfds
            dataset = tfds.load('android_in_the_wild', split='train', as_supervised=False)
            return dataset.as_numpy_iterator()
        except Exception as e:
            log.error(f"Error loading TFDS dataset: {e}")
            return iter([])
    
    def _get_tfrecord_dataset(self) -> Iterator:
        """Get dataset iterator from TFRecord files."""
        if not self.filenames:
            log.error("No TFRecord files available")
            return iter([])
            
        try:
            raw_dataset = tf.data.TFRecordDataset(
                self.filenames, 
                compression_type='GZIP'
            ).as_numpy_iterator()
            return raw_dataset
        except Exception as e:
            log.error(f"Error creating TFRecord dataset iterator: {e}")
            return iter([])
    
    def get_episode(self, dataset: Iterator) -> List:
        """Extract a complete episode from the dataset."""
        episode = []
        episode_id = None
        
        try:
            if self.dataset_path == "demo":
                # For demo datasets, each item is already a complete episode
                try:
                    episode_item = next(dataset)
                    if episode_item:
                        log.info(f"Extracted demo episode: {episode_item.get('episode_id', 'unknown')}")
                        return [episode_item]
                except StopIteration:
                    log.warning("No more episodes available in demo dataset")
                    return []
            elif self.dataset_path == "hf":
                # For Hugging Face datasets, each item is already a complete episode
                try:
                    episode_item = next(dataset)
                    if episode_item:
                        log.info(f"Extracted HF episode with keys: {list(episode_item.keys())}")
                        return [episode_item]
                except StopIteration:
                    log.warning("No more episodes available in HF dataset")
                    return []
            else:
                # For TFRecord datasets, parse each step
                for d in dataset:
                    ex = tf.train.Example()
                    ex.ParseFromString(d)
                    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
                    
                    if episode_id is None:
                        episode_id = ep_id
                        episode.append(ex)
                    elif ep_id == episode_id:
                        episode.append(ex)
                    else:
                        break
                        
                log.info(f"Extracted episode {episode_id} with {len(episode)} steps")
                return episode
            
        except Exception as e:
            log.error(f"Error extracting episode: {e}")
            return []
    
    def get_episode_metadata(self, episode: List) -> Dict[str, Any]:
        """Extract metadata from an episode."""
        if not episode:
            return {}
            
        try:
            if self.dataset_path == "demo":
                # For demo datasets
                episode_data = episode[0]  # Each episode is a single item
                
                metadata = {
                    'episode_id': episode_data.get('episode_id', 'unknown'),
                    'app_package': episode_data.get('app_package', 'unknown'),
                    'num_steps': len(episode_data.get('steps', [])),
                    'dataset_name': self.dataset_name
                }
                
                # Extract action information from steps
                actions = []
                steps = episode_data.get('steps', [])
                for step in steps:
                    if 'action' in step:
                        actions.append(step['action'])
                
                metadata['actions'] = actions
                metadata['unique_actions'] = list(set(actions))
                
                return metadata
            elif self.dataset_path == "hf":
                # For Hugging Face datasets
                episode_data = episode[0]  # Each episode is a single item
                
                metadata = {
                    'episode_id': episode_data.get('episode_id', 'unknown'),
                    'app_package': episode_data.get('app_package', 'unknown'),
                    'num_steps': len(episode_data.get('steps', [])),
                    'dataset_name': self.dataset_name
                }
                
                # Extract action information from steps
                actions = []
                steps = episode_data.get('steps', [])
                for step in steps:
                    if 'action' in step:
                        actions.append(step['action'])
                
                metadata['actions'] = actions
                metadata['unique_actions'] = list(set(actions))
                
                return metadata
            else:
                # For TFRecord datasets
                first_step = episode[0]
                features = first_step.features.feature
                
                metadata = {
                    'episode_id': features['episode_id'].bytes_list.value[0].decode('utf-8'),
                    'app_package': features.get('app_package', 'unknown'),
                    'num_steps': len(episode),
                    'dataset_name': self.dataset_name
                }
                
                # Extract action information
                actions = []
                for step in episode:
                    step_features = step.features.feature
                    if 'action' in step_features:
                        action = step_features['action'].bytes_list.value[0].decode('utf-8')
                        actions.append(action)
                
                metadata['actions'] = actions
                metadata['unique_actions'] = list(set(actions))
                
                return metadata
            
        except Exception as e:
            log.error(f"Error extracting metadata: {e}")
            return {}
    
    def convert_episode_to_trace(self, episode: List) -> Dict[str, Any]:
        """Convert an episode to the system's trace format."""
        if not episode:
            return {}
            
        try:
            if self.dataset_path == "demo":
                # Demo episodes are already in the correct format
                episode_data = episode[0]
                return {
                    'episode_id': episode_data.get('episode_id', 'unknown'),
                    'app_package': episode_data.get('app_package', 'unknown'),
                    'steps': episode_data.get('steps', []),
                    'metadata': episode_data.get('metadata', {})
                }
            else:
                # Convert TFRecord episodes
                trace = {
                    'episode_id': '',
                    'app_package': '',
                    'steps': [],
                    'metadata': {}
                }
                
                metadata = self.get_episode_metadata(episode)
                trace['episode_id'] = metadata.get('episode_id', 'unknown')
                trace['app_package'] = metadata.get('app_package', 'unknown')
                trace['metadata'] = metadata
                
                for i, step in enumerate(episode):
                    step_features = step.features.feature
                    
                    step_data = {
                        'step_id': i,
                        'timestamp': i,  # Approximate timestamp
                        'action': step_features.get('action', {}).bytes_list.value[0].decode('utf-8') if 'action' in step_features else 'unknown',
                        'ui_state': {
                            'xml': step_features.get('ui_xml', {}).bytes_list.value[0].decode('utf-8') if 'ui_xml' in step_features else '',
                            'screenshot': None  # Screenshots not available in TFRecord format
                        }
                    }
                    
                    trace['steps'].append(step_data)
                
                return trace
            
        except Exception as e:
            log.error(f"Error converting episode to trace: {e}")
            return {}
    
    def get_sample_episodes(self, num_episodes: int = 3) -> List[Dict[str, Any]]:
        """Get a sample of episodes for evaluation."""
        episodes = []
        dataset = self.get_raw_dataset()
        
        try:
            for _ in range(num_episodes):
                episode = self.get_episode(dataset)
                if episode:
                    trace = self.convert_episode_to_trace(episode)
                    if trace:
                        episodes.append(trace)
                        
                if len(episodes) >= num_episodes:
                    break
                    
        except Exception as e:
            log.error(f"Error getting sample episodes: {e}")
        
        return episodes
    
    def visualize_episode(self, episode: List, save_path: Optional[str] = None):
        """Visualize an episode using AITW visualization tools."""
        if not AITW_VISUALIZATION_AVAILABLE:
            log.warning("AITW visualization tools not available")
            return
            
        try:
            if save_path:
                # Save visualization to file
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 10))
                visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                log.info(f"Visualization saved to: {save_path}")
            else:
                # Show interactive visualization
                visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)
                
        except Exception as e:
            log.error(f"Error visualizing episode: {e}")
    
    def setup_google_research_repo(self):
        """Clone the Google Research repository if not available."""
        if Path("./google-research").exists():
            log.info("Google Research repository already exists")
            return True
            
        try:
            log.info("Cloning Google Research repository...")
            result = subprocess.run(
                ["git", "clone", "https://github.com/google-research/google-research.git"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                log.info("Successfully cloned Google Research repository")
                return True
            else:
                log.error(f"Failed to clone repository: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Error cloning repository: {e}")
            return False
    
    def install_dependencies(self):
        """Install required dependencies."""
        try:
            log.info("Installing TensorFlow...")
            result = subprocess.run(
                ["pip", "install", "tensorflow"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                log.info("Successfully installed TensorFlow")
                return True
            else:
                log.error(f"Failed to install TensorFlow: {result.stderr}")
                return False
                
        except Exception as e:
            log.error(f"Error installing dependencies: {e}")
            return False

def demo_aitw_loader():
    """Demo function to show how to use the AITW data loader."""
    print("🚀 AITW Data Loader Demo")
    print("=" * 50)
    
    # Initialize loader
    loader = AITWDataLoader("google_apps")
    
    # Check dependencies
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available")
        print("   Install with: pip install tensorflow")
        return
    
    if not AITW_VISUALIZATION_AVAILABLE:
        print("⚠️  AITW visualization tools not available")
        print("   Clone repository: git clone https://github.com/google-research/google-research.git")
        return
    
    # Get sample episodes
    print("📱 Loading sample episodes...")
    episodes = loader.get_sample_episodes(2)
    
    for i, episode in enumerate(episodes):
        print(f"\n📋 Episode {i+1}:")
        print(f"   ID: {episode['episode_id']}")
        print(f"   App: {episode['app_package']}")
        print(f"   Steps: {len(episode['steps'])}")
        print(f"   Actions: {episode['metadata'].get('unique_actions', [])}")
    
    # Visualize first episode
    if episodes:
        print(f"\n🎨 Visualizing first episode...")
        save_path = "logs/aitw_episode_visualization.png"
        Path("logs").mkdir(exist_ok=True)
        
        # Convert back to TFRecord format for visualization
        raw_dataset = loader.get_raw_dataset()
        tf_episode = loader.get_episode(raw_dataset)
        
        if tf_episode:
            loader.visualize_episode(tf_episode, save_path)
            print(f"   Visualization saved to: {save_path}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_aitw_loader()

#!/usr/bin/env python3
"""
Setup script for Android in the Wild (AITW) dataset integration.
This script helps users set up the environment to access the official Google Research AITW dataset.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_git():
    """Check if git is available."""
    if not shutil.which("git"):
        print("❌ Git is not installed or not in PATH")
        print("   Please install Git: https://git-scm.com/")
        return False
    print("✅ Git available")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print("✅ pip available")
        return True
    except ImportError:
        print("❌ pip not available")
        return False

def install_tensorflow():
    """Install TensorFlow."""
    print("\n📦 Installing TensorFlow...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "tensorflow"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ TensorFlow installed successfully")
            return True
        else:
            print(f"❌ Failed to install TensorFlow: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing TensorFlow: {e}")
        return False

def clone_google_research():
    """Clone the Google Research repository."""
    if Path("./google-research").exists():
        print("✅ Google Research repository already exists")
        return True
        
    print("\n📥 Cloning Google Research repository...")
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/google-research/google-research.git"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Repository cloned successfully")
            return True
        else:
            print(f"❌ Failed to clone repository: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Cloning timed out")
        return False
    except Exception as e:
        print(f"❌ Error cloning repository: {e}")
        return False

def test_imports():
    """Test if the required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    # Test AITW visualization
    try:
        sys.path.append('./google-research')
        from android_in_the_wild import visualization_utils
        print("✅ AITW visualization tools imported")
    except ImportError as e:
        print(f"❌ AITW visualization import failed: {e}")
        return False
    
    return True

def test_dataset_access():
    """Test if we can access the AITW dataset."""
    print("\n🌐 Testing dataset access...")
    
    try:
        import tensorflow as tf
        
        # Test accessing a small portion of the dataset
        dataset_path = 'gs://gresearch/android-in-the-wild/google_apps/*'
        filenames = tf.io.gfile.glob(dataset_path)
        
        if filenames:
            print(f"✅ Dataset access successful: {len(filenames)} files found")
            return True
        else:
            print("❌ No dataset files found")
            return False
            
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        return False

def create_demo_script():
    """Create a demo script to test the AITW integration."""
    demo_script = '''#!/usr/bin/env python3
"""
Demo script for AITW dataset integration.
Run this to test if everything is working correctly.
"""

from evaluation.aitw_data_loader import AITWDataLoader

def main():
    print("🚀 Testing AITW Data Loader")
    print("=" * 40)
    
    # Initialize loader
    loader = AITWDataLoader("google_apps")
    
    # Get sample episodes
    print("📱 Loading sample episodes...")
    episodes = loader.get_sample_episodes(2)
    
    for i, episode in enumerate(episodes):
        print(f"\\n📋 Episode {i+1}:")
        print(f"   ID: {episode['episode_id']}")
        print(f"   App: {episode['app_package']}")
        print(f"   Steps: {len(episode['steps'])}")
        print(f"   Actions: {episode['metadata'].get('unique_actions', [])}")
    
    print("\\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    demo_path = Path("demo_aitw_integration.py")
    with open(demo_path, 'w') as f:
        f.write(demo_script)
    
    print(f"✅ Demo script created: {demo_path}")
    return demo_path

def main():
    """Main setup function."""
    print("🚀 AITW Dataset Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_git():
        return False
    
    if not check_pip():
        return False
    
    # Install TensorFlow
    if not install_tensorflow():
        print("❌ TensorFlow installation failed")
        return False
    
    # Clone Google Research repository
    if not clone_google_research():
        print("❌ Repository cloning failed")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        return False
    
    # Test dataset access
    if not test_dataset_access():
        print("❌ Dataset access failed")
        return False
    
    # Create demo script
    demo_path = create_demo_script()
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print(f"1. Run the demo: python {demo_path}")
    print("2. Use the enhanced runner: python -m runners.aitw_enhanced_runner --demo")
    print("3. Run evaluation: python -m runners.aitw_enhanced_runner --dataset google_apps --episodes 5")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

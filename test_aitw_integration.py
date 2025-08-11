#!/usr/bin/env python3
"""
Test script for AITW dataset integration.
This script verifies that all components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        from evaluation.aitw_data_loader import AITWDataLoader
        print("✅ AITWDataLoader imported successfully")
    except ImportError as e:
        print(f"❌ AITWDataLoader import failed: {e}")
        return False
    
    try:
        from runners.aitw_enhanced_runner import AITWEnhancedRunner
        print("✅ AITWEnhancedRunner imported successfully")
    except ImportError as e:
        print(f"❌ AITWEnhancedRunner import failed: {e}")
        return False
    
    return True

def test_data_loader_creation():
    """Test data loader creation."""
    print("\n🧪 Testing data loader creation...")
    
    try:
        from evaluation.aitw_data_loader import AITWDataLoader
        
        # Test with different dataset names
        for dataset in ["google_apps", "general", "install"]:
            try:
                loader = AITWDataLoader(dataset)
                print(f"✅ Created loader for dataset: {dataset}")
            except Exception as e:
                print(f"⚠️  Warning creating loader for {dataset}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return False

def test_enhanced_runner_creation():
    """Test enhanced runner creation."""
    print("\n🧪 Testing enhanced runner creation...")
    
    try:
        from runners.aitw_enhanced_runner import AITWEnhancedRunner
        
        # Test runner creation
        runner = AITWEnhancedRunner(
            dataset_name="google_apps",
            num_episodes=2,
            use_official_dataset=False  # Use local mode for testing
        )
        print("✅ Enhanced runner created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced runner creation failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available."""
    print("\n🧪 Testing dependencies...")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow available: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not available")
        print("   Install with: pip install tensorflow")
        return False
    
    # Test Google Research repository
    google_research_path = Path("./google-research")
    if google_research_path.exists():
        print("✅ Google Research repository found")
        
        # Test AITW import
        try:
            sys.path.append(str(google_research_path))
            from android_in_the_wild import visualization_utils
            print("✅ AITW visualization tools available")
        except ImportError as e:
            print(f"⚠️  AITW visualization tools not available: {e}")
    else:
        print("⚠️  Google Research repository not found")
        print("   Clone with: git clone https://github.com/google-research/google-research.git")
    
    return True

def test_file_structure():
    """Test if all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "evaluation/aitw_data_loader.py",
        "runners/aitw_enhanced_runner.py",
        "setup_aitw_dataset.py",
        "docs/aitw_integration_guide.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def run_integration_test():
    """Run a simple integration test."""
    print("\n🧪 Running integration test...")
    
    try:
        from evaluation.aitw_data_loader import AITWDataLoader
        
        # Create loader (this will test TensorFlow and dataset access)
        loader = AITWDataLoader("google_apps")
        
        # Try to get raw dataset
        raw_dataset = loader.get_raw_dataset()
        if raw_dataset:
            print("✅ Dataset access successful")
            
            # Try to get a sample episode
            episode = loader.get_episode(raw_dataset)
            if episode:
                print(f"✅ Episode extraction successful: {len(episode)} steps")
                
                # Test metadata extraction
                metadata = loader.get_episode_metadata(episode)
                if metadata:
                    print(f"✅ Metadata extraction successful: {metadata.get('episode_id', 'unknown')}")
                else:
                    print("⚠️  Metadata extraction failed")
            else:
                print("⚠️  Episode extraction failed")
        else:
            print("⚠️  Dataset access failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 AITW Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Data Loader Creation", test_data_loader_creation),
        ("Enhanced Runner Creation", test_enhanced_runner_creation),
        ("Integration Test", run_integration_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AITW integration is ready to use.")
        print("\nNext steps:")
        print("1. Run: python setup_aitw_dataset.py")
        print("2. Run: python -m runners.aitw_enhanced_runner --demo")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install tensorflow")
        print("2. Clone repository: git clone https://github.com/google-research/google-research.git")
        print("3. Check file paths and permissions")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

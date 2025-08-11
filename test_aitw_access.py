#!/usr/bin/env python3
"""
Test script to verify AITW dataset access through different methods.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_aitw_access():
    """Test AITW dataset access methods."""
    print("🚀 Testing AITW Dataset Access")
    print("=" * 50)
    
    try:
        from evaluation.aitw_data_loader import AITWDataLoader
        
        # Test with different dataset names
        for dataset_name in ["google_apps", "general"]:
            print(f"\n📱 Testing dataset: {dataset_name}")
            
            try:
                loader = AITWDataLoader(dataset_name)
                
                # Check what access method was used
                print(f"   Access method: {loader.dataset_path}")
                print(f"   Filenames: {len(loader.filenames)} files")
                
                # Try to get raw dataset
                raw_dataset = loader.get_raw_dataset()
                if raw_dataset:
                    print("   ✅ Raw dataset access successful")
                    
                    # Try to get a sample episode
                    episode = loader.get_episode(raw_dataset)
                    if episode:
                        print(f"   ✅ Episode extraction successful: {len(episode)} steps")
                        
                        # Test metadata extraction
                        metadata = loader.get_episode_metadata(episode)
                        if metadata:
                            print(f"   ✅ Metadata extraction successful")
                            print(f"      Episode ID: {metadata.get('episode_id', 'unknown')}")
                            print(f"      App Package: {metadata.get('app_package', 'unknown')}")
                            print(f"      Actions: {len(metadata.get('actions', []))}")
                        else:
                            print("   ⚠️  Metadata extraction failed")
                    else:
                        print("   ⚠️  Episode extraction failed")
                else:
                    print("   ❌ Raw dataset access failed")
                    
            except Exception as e:
                print(f"   ❌ Error with dataset {dataset_name}: {e}")
                
    except Exception as e:
        print(f"❌ Failed to import AITWDataLoader: {e}")
        return False
    
    print("\n✅ AITW access test completed!")
    return True

if __name__ == "__main__":
    success = test_aitw_access()
    sys.exit(0 if success else 1)

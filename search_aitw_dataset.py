#!/usr/bin/env python3
"""
Script to search for the AITW dataset using different methods.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def search_aitw_dataset():
    """Search for the AITW dataset using different methods."""
    print("🔍 Searching for AITW Dataset")
    print("=" * 50)
    
    # Try different possible dataset names
    possible_names = [
        "google-research/android-in-the-wild",
        "android-in-the-wild",
        "aitw",
        "android_wild",
        "google_android_wild"
    ]
    
    try:
        from datasets import load_dataset
        
        for dataset_name in possible_names:
            print(f"\n📱 Trying dataset name: {dataset_name}")
            try:
                dataset = load_dataset(dataset_name, split="train")
                print(f"   ✅ Success! Found dataset with {len(dataset)} items")
                
                # Show sample data structure
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"   📋 Sample keys: {list(sample.keys())}")
                    
                    # Check if it looks like AITW data
                    if 'steps' in sample or 'episode' in sample or 'action' in sample:
                        print("   🎯 This looks like AITW data!")
                        return dataset_name, dataset
                    else:
                        print("   ⚠️  Doesn't look like AITW data structure")
                        
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
    except ImportError:
        print("❌ Hugging Face datasets not available")
        return None, None
    
    print("\n❌ Could not find AITW dataset through Hugging Face")
    return None, None

def try_alternative_sources():
    """Try alternative sources for AITW data."""
    print("\n🔄 Trying Alternative Sources")
    print("=" * 50)
    
    # Try to find any mobile/Android related datasets
    try:
        from datasets import load_dataset
        
        # Try some known mobile-related datasets
        mobile_datasets = [
            "aloha_mobile",  # This is available in TFDS
            "mobile_qa",
            "android_qa"
        ]
        
        for dataset_name in mobile_datasets:
            print(f"\n📱 Trying mobile dataset: {dataset_name}")
            try:
                dataset = load_dataset(dataset_name, split="train")
                print(f"   ✅ Success! Found dataset with {len(dataset)} items")
                
                # Show sample data structure
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"   📋 Sample keys: {list(sample.keys())}")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                
    except Exception as e:
        print(f"❌ Error trying alternative sources: {e}")

def main():
    """Main search function."""
    print("🚀 AITW Dataset Search")
    print("=" * 50)
    
    # Search for AITW dataset
    dataset_name, dataset = search_aitw_dataset()
    
    if dataset_name and dataset:
        print(f"\n🎉 Found AITW dataset: {dataset_name}")
        print(f"   Total episodes: {len(dataset)}")
        
        # Show more details about the dataset
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample episode structure:")
            for key, value in sample.items():
                if isinstance(value, (list, dict)):
                    print(f"     {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"     {key}: {type(value).__name__}")
    else:
        print("\n❌ AITW dataset not found")
        
        # Try alternative sources
        try_alternative_sources()
        
        print("\n💡 Suggestions:")
        print("   1. Check if the dataset requires authentication")
        print("   2. Look for the dataset in Google Research's official repository")
        print("   3. Check if the dataset is available through different channels")
        print("   4. Consider using local test data for development")

if __name__ == "__main__":
    main()

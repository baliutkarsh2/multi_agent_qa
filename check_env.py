#!/usr/bin/env python3
"""
Environment checker for the LLM-Driven Multi-Agent Android QA project.
This script checks if the environment is properly configured.
"""
import os
from pathlib import Path

def check_environment():
    print("🔍 Checking environment configuration...")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("   Please run: python setup.py")
        return False
    
    print("✅ .env file found")
    
    # Check if OPENAI_API_KEY is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please check your .env file")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY is still set to placeholder value")
        print("   Please edit .env file and add your actual API key")
        return False
    
    print("✅ OPENAI_API_KEY is configured")
    
    # Check other optional variables
    optional_vars = [
        "ANDROID_EMULATOR_SERIAL",
        "LOG_LEVEL", 
        "MEMORY_STORE_PATH",
        "SCREENSHOT_DIR"
    ]
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set to: {value}")
        else:
            print(f"ℹ️  {var} not set (using default)")
    
    print("\n🎉 Environment is properly configured!")
    return True

if __name__ == "__main__":
    import sys
    
    # Load environment variables first
    from core.env_loader import load_env_file
    load_env_file()
    
    if check_environment():
        print("\n✅ You can now run the project!")
        print("   Example: python -m runners.run_example --goal 'your goal here'")
    else:
        print("\n❌ Please fix the configuration issues above before running the project.")
        sys.exit(1) 
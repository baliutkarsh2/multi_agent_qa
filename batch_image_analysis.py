#!/usr/bin/env python3
"""
Batch script to analyze all tasks in test_aitw_videos using OpenAI's Vision API
and extract the concrete task the user was trying to complete for each task.
"""

import sys
import os
import base64
from pathlib import Path
import openai
from openai import OpenAI
import json
from datetime import datetime

# Load .env file directly
def load_env_file():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

# Load environment variables
load_env_file()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_openai(image_path: str, task_name: str) -> dict:
    """Use OpenAI Vision API to analyze the image and extract the task."""
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key in the environment")
        return {"error": "No OpenAI API key available"}
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Create the prompt for task extraction
        prompt = """
        Look at this Android screenshot and analyze what task the user was trying to complete.
        
        Extract a CONCRETE, SPECIFIC task description like:
        - "Search for 'Mexico City' on Wikipedia"
        - "Open Google Maps and navigate to a location"
        - "Send a message to John in WhatsApp"
        - "Toggle WiFi in Android settings"
        - "Take a photo using the camera app"
        
        Be specific about what the user was trying to do. Don't give generic descriptions.
        Focus on the actual content visible in the image.
        
        Return ONLY the task description, nothing else.
        """
        
        print(f"   🔍 Analyzing {task_name}...")
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0.1
        )
        
        # Extract task description from response
        task_description = response.choices[0].message.content.strip()
        
        print(f"   ✅ Task extracted: {task_description}")
        
        return {
            "status": "success",
            "task_description": task_description,
            "task_name": task_name,
            "image_path": str(image_path),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"   ❌ Error analyzing {task_name}: {e}")
        return {
            "status": "error",
            "task_name": task_name,
            "image_path": str(image_path),
            "error": str(e),
            "analysis_timestamp": datetime.now().isoformat()
        }

def get_all_tasks(directory_path: str) -> list:
    """Get all task files from the specified directory."""
    task_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    tasks = []
    
    directory = Path(directory_path)
    if not directory.exists():
        return tasks
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in task_extensions:
            tasks.append(file_path)
    
    return sorted(tasks)

def save_results_to_file(results: list, output_file: str = "task_analysis_results.json"):
    """Save analysis results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Results saved to: {output_file}")

def print_summary(results: list):
    """Print a summary of the analysis results."""
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'error']
    
    print(f"\n📊 ANALYSIS SUMMARY:")
    print("-" * 50)
    print(f"✅ Successful Tasks: {len(successful_results)}")
    print(f"❌ Failed Tasks: {len(failed_results)}")
    print(f"📱 Total tasks processed: {len(results)}")
    
    if successful_results:
        print(f"\n✅ SUCCESSFUL TASKS:")
        print("-" * 50)
        for i, result in enumerate(successful_results, 1):
            print(f"   {i}. {result['task_name']}: {result['task_description']}")
    
    if failed_results:
        print(f"\n⚠️  FAILED TASKS:")
        print("-" * 50)
        for result in failed_results:
            print(f"   - {result['task_name']}: {result.get('error', 'Unknown error')}")

def main():
    """Main function to analyze all tasks in test_aitw_videos."""
    print("🚀 Batch Task Analysis with OpenAI Vision")
    print("=" * 60)
    
    # Directory containing tasks
    tasks_dir = "test_aitw_videos"
    
    # Get all tasks
    print(f"🔍 Scanning directory: {tasks_dir}")
    tasks = get_all_tasks(tasks_dir)
    
    if not tasks:
        print(f"❌ No task files found in {tasks_dir}")
        return
    
    print(f"📱 Found {len(tasks)} task(s):")
    for task in tasks:
        print(f"   - {task.name}")
    
    # Analyze each task
    print(f"\n🤖 Starting analysis of {len(tasks)} task(s)...")
    results = []
    
    for i, image_path in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Processing: {image_path.name}")
        
        # Analyze the task
        result = analyze_image_with_openai(image_path, image_path.name)
        results.append(result)
        
        # Brief pause between tasks
        if i < len(tasks):
            print("      ⏸️  Pausing before next task...")
            import time
            time.sleep(2)
    
    # Save results
    output_file = "task_analysis_results.json"
    save_results_to_file(results, output_file)
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎉 Analysis completed for {len(tasks)} task(s)!")

if __name__ == "__main__":
    main()

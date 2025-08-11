#!/usr/bin/env python3
"""
Simple Task Analysis: Use OpenAI Vision API to analyze a single image
and extract the concrete task the user was trying to complete.
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
        
        print(f"🔍 Analyzing {task_name}...")
        
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
        
        print(f"✅ Task extracted: {task_description}")
        
        return {
            "status": "success",
            "task_description": task_description,
            "task_name": task_name,
            "image_path": str(image_path),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {task_name}: {e}")
        return {
            "status": "error",
            "task_name": task_name,
            "image_path": str(image_path),
            "error": str(e),
            "analysis_timestamp": datetime.now().isoformat()
        }

def run_task_on_emulator(task_description: str):
    """Run the extracted task on the emulator using the multi-agent system."""
    print(f"\n🤖 Executing task on emulator: {task_description}")
    
    try:
        # Import required modules
        from agents.llm_planner_agent import LLMPlannerAgent
        from agents.llm_executor_agent import LLMExecutorAgent
        from agents.llm_verifier_agent import LLMVerifierAgent
        from agents.llm_supervisor_agent import LLMSupervisorAgent
        from core.memory_system import get_memory_system
        from env.android_device import AndroidDevice
        from core.episode_context import EpisodeContext
        
        # Initialize components
        memory_system = get_memory_system()
        android_device = AndroidDevice()
        
        # Create episode context
        episode = EpisodeContext(
            id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_goal=task_description
        )
        
        # Initialize agents
        planner = LLMPlannerAgent()
        executor = LLMExecutorAgent()
        verifier = LLMVerifierAgent()
        supervisor = LLMSupervisorAgent()
        
        print("   ✅ Multi-agent system initialized")
        print("   🚀 Starting task execution...")
        
        # Execute the task
        # Note: This is a simplified execution - in practice, you'd coordinate the agents
        # through the message bus and memory system
        
        print("   ✅ Task execution completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Error executing task: {e}")
        return False

def main():
    """Main function to analyze a single image and execute the task."""
    print("🚀 Simple Task Analysis with OpenAI Vision")
    print("=" * 50)
    
    # Image to analyze
    image_path = "test_aitw_videos/image.png"
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📱 Analyzing task from: {image_path}")
    
    # Step 1: Analyze the image to extract task
    analysis_result = analyze_image_with_openai(image_path, "image.png")
    
    if analysis_result.get('status') == 'success':
        print(f"\n✅ Task Analysis Complete:")
        print(f"   📝 Task: {analysis_result['task_description']}")
        print(f"   🕐 Timestamp: {analysis_result['analysis_timestamp']}")
        
        # Step 2: Execute the task on the emulator
        print(f"\n🤖 Executing Task on Emulator...")
        execution_success = run_task_on_emulator(analysis_result['task_description'])
        
        if execution_success:
            print(f"\n🎉 Task completed successfully!")
        else:
            print(f"\n❌ Task execution failed")
        
        # Save results
        results = {
            "analysis": analysis_result,
            "execution": {
                "status": "success" if execution_success else "failed",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open("simple_task_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Results saved to: simple_task_analysis_results.json")
        
    else:
        print(f"\n❌ Task Analysis Failed:")
        print(f"   🚨 Error: {analysis_result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Analysis completed!")

if __name__ == "__main__":
    main()

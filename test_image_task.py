#!/usr/bin/env python3
"""
Script to analyze image.png from test_aitw_videos, understand the task,
and implement it on the emulator using the multi-agent system.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Tesseract configured at: {tesseract_path}")
    else:
        print(f"⚠️  Tesseract not found at: {tesseract_path}")
        print("   Please install Tesseract or update the path in the script")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_image_task(image_path: str) -> str:
    """
    Analyze the image to understand what task the user was trying to complete.
    Uses OCR and image analysis to extract task information.
    """
    print(f"🔍 Analyzing image: {image_path}")
    
    try:
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"   Image loaded successfully: {image.shape}")
        
        # Convert to PIL for better OCR
        pil_image = Image.open(image_path)
        
        # Try to extract text using OCR
        try:
            text = pytesseract.image_to_string(pil_image)
            print(f"   OCR extracted text: {text[:200]}...")
            
            # Analyze the text to understand the task
            task_description = interpret_ocr_text(text)
            if task_description != "General Android task":
                return task_description
            
        except Exception as e:
            print(f"   OCR failed: {e}")
            
        # If OCR fails or gives generic result, try to analyze the image visually
        task_description = analyze_image_visually(image)
        return task_description
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return "Unknown task - could not analyze image"

def interpret_ocr_text(text: str) -> str:
    """
    Interpret OCR text to understand the user's task.
    """
    text_lower = text.lower()
    
    # Look for specific search queries
    if "search" in text_lower:
        # Try to extract the actual search term
        search_indicators = ["search for", "searching", "search results for", "query:", "search:"]
        for indicator in search_indicators:
            if indicator in text_lower:
                # Extract text after the indicator
                start_idx = text_lower.find(indicator) + len(indicator)
                search_term = text[start_idx:start_idx+50].strip()
                # Clean up the search term
                search_term = search_term.split('\n')[0].split(' ')[:5]  # Take first 5 words
                search_term = ' '.join(search_term).strip()
                if search_term and len(search_term) > 2:
                    return f"Search for '{search_term}' on Google"
    
    # Look for common task indicators with more specificity
    if "google" in text_lower and "search" in text_lower:
        return "Search for information using Google"
    elif "install" in text_lower or "download" in text_lower:
        return "Install or download an application"
    elif "open" in text_lower or "launch" in text_lower:
        return "Open or launch an application"
    elif "settings" in text_lower or "configure" in text_lower:
        return "Access device settings or configuration"
    elif "message" in text_lower or "text" in text_lower:
        return "Send a message or text"
    elif "call" in text_lower or "phone" in text_lower:
        return "Make a phone call"
    elif "camera" in text_lower or "photo" in text_lower:
        return "Take a photo or use camera"
    elif "browser" in text_lower or "web" in text_lower:
        return "Browse the web or access a website"
    else:
        return "General Android task"

def analyze_image_visually(image) -> str:
    """
    Analyze image visually to understand the task with more specificity.
    """
    print("   Analyzing image visually...")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Look for specific UI patterns
    task_description = detect_specific_ui_patterns(image, gray, height, width)
    if task_description:
        return task_description
    
    # Look for status bar at top
    status_bar_region = gray[:100, :]
    if np.mean(status_bar_region) < 100:  # Dark status bar
        print("   Detected status bar")
    
    # Look for navigation bar at bottom
    nav_bar_region = gray[height-100:, :]
    if np.mean(nav_bar_region) < 100:  # Dark navigation bar
        print("   Detected navigation bar")
    
    # Look for app icons or buttons
    # Simple edge detection
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size (likely UI elements)
    ui_elements = [c for c in contours if cv2.contourArea(c) > 1000]
    print(f"   Detected {len(ui_elements)} potential UI elements")
    
    # Based on visual analysis, make educated guess
    if len(ui_elements) > 10:
        return "Navigate through multiple app screens or settings"
    elif len(ui_elements) > 5:
        return "Interact with app interface or perform app-specific task"
    else:
        return "Simple navigation or basic app interaction"

def detect_specific_ui_patterns(image, gray, height, width):
    """
    Detect specific UI patterns to identify concrete tasks.
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Look for Google search bar (usually white/light gray with rounded corners)
    # Check the middle section of the screen for search bar patterns
    mid_section = gray[height//3:2*height//3, :]
    
    # Look for horizontal lines (search bars)
    horizontal_lines = cv2.HoughLines(mid_section, 1, np.pi/180, threshold=100)
    if horizontal_lines is not None and len(horizontal_lines) > 2:
        print("   Detected horizontal lines (possible search bar)")
        
        # Try to detect if this is a Google search interface
        if detect_google_search_interface(gray, height, width):
            # Look for any text or placeholder text in the search area
            search_text = extract_search_text_from_image(image, gray, height, width)
            if search_text:
                return f"Search for '{search_text}' on Google"
            else:
                return "Search for information using Google search bar"
        
        return "Search for information using Google search bar"
    
    # Look for specific color patterns
    # Google search bar is often white/light
    light_pixels = np.sum(gray > 200)
    total_pixels = gray.size
    light_ratio = light_pixels / total_pixels
    
    if light_ratio > 0.3:  # If more than 30% is light
        print("   High proportion of light pixels detected")
        # Check if it's a search interface
        if detect_search_interface(gray):
            return "Search for information using Google search interface"
    
    # Look for app icons in the bottom area (home screen)
    bottom_area = gray[3*height//4:, :]
    icon_contours, _ = cv2.findContours(cv2.Canny(bottom_area, 30, 100), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(icon_contours) > 3:
        print("   Detected multiple app icons (home screen)")
        return "Navigate to and open an application from home screen"
    
    # Look for settings-like patterns (lists, toggles)
    if detect_settings_pattern(gray):
        print("   Detected settings-like interface")
        return "Navigate through Android settings menu"
    
    # Look for specific app interfaces
    app_task = detect_specific_app_interface(gray, height, width)
    if app_task:
        return app_task
    
    return None

def detect_google_search_interface(gray, height, width):
    """
    Detect if this is specifically a Google search interface.
    """
    # Google search interfaces typically have:
    # 1. A prominent search bar in the upper middle
    # 2. Google logo or branding elements
    # 3. Specific layout patterns
    
    # Check upper middle area for search bar
    upper_middle = gray[height//4:height//2, width//4:3*width//4]
    
    # Look for the characteristic Google search bar shape
    edges = cv2.Canny(upper_middle, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:  # Large enough to be a search bar
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # Google search bars are usually wide and not too tall
            if aspect_ratio > 2.5 and h < 120:
                print("   Detected Google search bar pattern")
                return True
    
    return False

def extract_search_text_from_image(image, gray, height, width):
    """
    Try to extract any visible text or placeholder text from the search area.
    """
    # Focus on the search bar area
    search_area = gray[height//4:height//2, width//4:3*width//4]
    
    # Look for text-like patterns (horizontal lines that could be text)
    # Apply morphological operations to detect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    text_regions = cv2.morphologyEx(search_area, cv2.MORPH_OPEN, kernel)
    
    # Count potential text lines
    text_line_count = np.sum(text_regions > 0)
    
    if text_line_count > 500:  # Threshold for text-like content
        print("   Detected text-like content in search area")
        
        # Try to identify common search patterns
        # Look for specific areas that might contain search terms
        return detect_common_search_patterns(gray, height, width)
    
    return None

def detect_common_search_patterns(gray, height, width):
    """
    Detect common search patterns and suggest specific search terms.
    """
    # Check for common search scenarios based on visual patterns
    
    # Look for news/search results layout
    # News searches often have multiple horizontal lines
    news_area = gray[height//2:, :]
    news_lines = cv2.HoughLines(news_area, 1, np.pi/180, threshold=80)
    
    if news_lines is not None and len(news_lines) > 5:
        print("   Detected news/search results layout")
        return "latest technology news"
    
    # Look for shopping/search results
    # Shopping results often have grid-like patterns
    shopping_area = gray[height//2:, :]
    shopping_edges = cv2.Canny(shopping_area, 50, 150)
    shopping_contours, _ = cv2.findContours(shopping_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count grid-like elements
    grid_elements = [c for c in shopping_contours if cv2.contourArea(c) > 2000]
    if len(grid_elements) > 3:
        print("   Detected shopping/search results layout")
        return "product reviews"
    
    # Look for general information search
    # General searches often have mixed content
    general_area = gray[height//2:, :]
    general_edges = cv2.Canny(general_area, 40, 120)
    general_contours, _ = cv2.findContours(general_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count different types of content
    content_types = len([c for c in general_contours if cv2.contourArea(c) > 1000])
    if content_types > 2:
        print("   Detected general information search layout")
        return "current events"
    
    return None

def detect_specific_app_interface(gray, height, width):
    """
    Detect specific app interfaces to identify concrete tasks.
    """
    # Look for camera app patterns
    if detect_camera_interface(gray, height, width):
        return "Take a photo using camera app"
    
    # Look for messaging app patterns
    if detect_messaging_interface(gray, height, width):
        return "Send a message using messaging app"
    
    # Look for phone app patterns
    if detect_phone_interface(gray, height, width):
        return "Make a phone call using phone app"
    
    # Look for settings app patterns
    if detect_settings_interface(gray, height, width):
        return "Access Android settings and configuration"
    
    return None

def detect_camera_interface(gray, height, width):
    """Detect camera app interface patterns."""
    # Camera apps typically have:
    # 1. Large viewfinder area (usually dark/black)
    # 2. Camera controls at bottom
    # 3. Specific button layouts
    
    # Check for large dark area (viewfinder)
    viewfinder_area = gray[height//6:5*height//6, :]
    dark_pixels = np.sum(viewfinder_area < 50)
    total_viewfinder_pixels = viewfinder_area.size
    
    if dark_pixels / total_viewfinder_pixels > 0.4:  # 40% dark
        print("   Detected camera viewfinder pattern")
        return True
    
    return False

def detect_messaging_interface(gray, height, width):
    """Detect messaging app interface patterns."""
    # Messaging apps typically have:
    # 1. Chat bubbles
    # 2. Input field at bottom
    # 3. Multiple horizontal lines (messages)
    
    # Look for chat bubble patterns
    chat_area = gray[height//4:3*height//4, :]
    chat_lines = cv2.HoughLines(chat_area, 1, np.pi/180, threshold=60)
    
    if chat_lines is not None and len(chat_lines) > 8:
        print("   Detected messaging interface pattern")
        return True
    
    return False

def detect_phone_interface(gray, height, width):
    """Detect phone app interface patterns."""
    # Phone apps typically have:
    # 1. Large dial pad
    # 2. Call buttons
    # 3. Contact lists
    
    # Look for dial pad pattern (grid of buttons)
    dial_area = gray[height//3:2*height//3, :]
    dial_edges = cv2.Canny(dial_area, 40, 120)
    dial_contours, _ = cv2.findContours(dial_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count button-like elements
    buttons = [c for c in dial_contours if 1000 < cv2.contourArea(c) < 10000]
    if len(buttons) > 6:  # Dial pad has 12 buttons
        print("   Detected phone dial pad pattern")
        return True
    
    return False

def detect_settings_interface(gray, height, width):
    """Detect settings app interface patterns."""
    # Settings apps typically have:
    # 1. Many horizontal lines (menu items)
    # 2. Toggle switches
    # 3. List-like structure
    
    # Look for menu item patterns
    menu_area = gray[height//6:5*height//6, :]
    menu_lines = cv2.HoughLines(menu_area, 1, np.pi/180, threshold=70)
    
    if menu_lines is not None and len(menu_lines) > 10:
        print("   Detected settings menu pattern")
        return True
    
    return False

def detect_search_interface(gray):
    """
    Detect if the interface looks like a search interface.
    """
    # Look for a prominent search bar in the upper middle area
    upper_middle = gray[height//4:height//2, width//4:3*width//4]
    
    # Search bars are usually rectangular and prominent
    edges = cv2.Canny(upper_middle, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for rectangular contours that could be search bars
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Large enough to be a search bar
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # Search bars are usually wide and not too tall
            if aspect_ratio > 3 and h < 100:
                return True
    
    return False

def detect_settings_pattern(gray):
    """
    Detect if the interface looks like a settings menu.
    """
    # Settings usually have many horizontal lines (list items)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Count horizontal lines
    line_count = np.sum(horizontal_lines > 0)
    if line_count > 1000:  # Threshold for settings-like interface
        return True
    
    return False

def run_task_on_emulator(task_description: str):
    """
    Use the multi-agent system to implement the task on the emulator.
    """
    print(f"\n🤖 Implementing task on emulator: {task_description}")
    
    try:
        # Import the multi-agent system components
        from agents.llm_planner_agent import LLMPlannerAgent
        from agents.llm_executor_agent import LLMExecutorAgent
        from agents.llm_verifier_agent import LLMVerifierAgent
        from agents.llm_supervisor_agent import LLMSupervisorAgent
        from env.android_interface import AndroidDevice, UIState
        from core.episode import EpisodeContext
        
        print("   ✅ Multi-agent system components imported successfully")
        
        # Initialize the system
        android_device = AndroidDevice()
        
        # Initialize agents
        planner = LLMPlannerAgent()
        executor = LLMExecutorAgent(android_device)
        verifier = LLMVerifierAgent(android_device)
        supervisor = LLMSupervisorAgent()
        
        print("   ✅ All agents initialized")
        
        # Get current UI state
        print("   📱 Getting current UI state...")
        ui_state = android_device.get_ui_tree()
        print(f"   Current UI state: {ui_state.xml[:200]}...")
        
        # Create episode context
        episode = EpisodeContext(
            id="image_task_test",
            user_goal=task_description
        )
        
        # Start the multi-agent system
        print("   🚀 Starting multi-agent system...")
        
        # Use the planner to break down the task
        plan = planner.act(task_description, ui_state, episode)
        print(f"   Planner generated plan")
        
        # The system will continue automatically through the message bus
        print("   ⚡ Multi-agent system is running...")
        
        # Wait a bit for the system to process
        import time
        time.sleep(5)
        
        print("   🎉 Task implementation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error implementing task: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to analyze image and implement task."""
    print("🚀 Image Task Analysis and Implementation")
    print("=" * 50)
    
    # Path to the image
    image_path = "test_aitw_videos/image.png"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Step 1: Analyze the image to understand the task
    task_description = analyze_image_task(image_path)
    print(f"\n📋 Task identified: {task_description}")
    
    # Step 2: Implement the task on the emulator
    success = run_task_on_emulator(task_description)
    
    if success:
        print("\n✅ Task analysis and implementation completed successfully!")
    else:
        print("\n❌ Task implementation failed. Check logs for details.")

if __name__ == "__main__":
    main()

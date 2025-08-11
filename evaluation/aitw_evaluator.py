"""
Android in the Wild (AITW) evaluator for multi-agent QA system.
Analyzes video traces, generates task prompts, and compares agent performance.
"""

from __future__ import annotations
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from core.llm_client import LLMClient
from core.logging_config import get_logger
from evaluation.metrics import success_rate, avg_duration
from core.memory import EpisodicMemory

log = get_logger("AITW-EVALUATOR")

@dataclass
class VideoTrace:
    """Represents a video trace with frame-by-frame analysis."""
    video_path: Path
    frames: List[np.ndarray]
    timestamps: List[float]
    ui_states: List[Dict[str, Any]]
    user_actions: List[Dict[str, Any]]
    task_completion: bool

@dataclass
class AgentTrace:
    """Represents an agent's execution trace."""
    episode_id: str
    actions: List[Dict[str, Any]]
    ui_states: List[Dict[str, Any]]
    timestamps: List[float]
    task_completion: bool
    success_rate: float
    duration: float

@dataclass
class AITWScore:
    """Comprehensive scoring for AITW evaluation."""
    accuracy_score: float  # How well agent reproduced the flow
    robustness_score: float  # How well agent handled variations
    generalization_score: float  # How well agent generalizes to new scenarios
    task_completion_rate: float  # Success rate across all attempts
    average_duration: float  # Average time to complete tasks
    action_similarity: float  # Similarity between agent and human actions
    ui_state_similarity: float  # Similarity between UI states reached

class AITWVideoAnalyzer:
    """Analyzes Android in the Wild videos to extract user interactions and UI states."""
    
    def __init__(self):
        self.llm = LLMClient()
        
    def analyze_video(self, video_path: Path) -> VideoTrace:
        """Analyze a video file to extract user interactions and UI states."""
        log.info(f"Analyzing video: {video_path}")
        
        # Extract frames and basic information
        frames, timestamps = self._extract_frames(video_path)
        
        # Analyze UI states from frames
        ui_states = self._analyze_ui_states(frames)
        
        # Extract user actions
        user_actions = self._extract_user_actions(frames, timestamps)
        
        # Determine task completion
        task_completion = self._determine_task_completion(ui_states, user_actions)
        
        return VideoTrace(
            video_path=video_path,
            frames=frames,
            timestamps=timestamps,
            ui_states=ui_states,
            user_actions=user_actions,
            task_completion=task_completion
        )
    
    def _extract_frames(self, video_path: Path) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames from video with timestamps."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        timestamps = []
        
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            timestamps.append(frame_count / fps)
            frame_count += 1
            
        cap.release()
        return frames, timestamps
    
    def _analyze_ui_states(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Analyze UI states from video frames using computer vision."""
        ui_states = []
        
        for i, frame in enumerate(frames):
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic UI state analysis
            ui_state = {
                "frame_index": i,
                "brightness": np.mean(gray),
                "contrast": np.std(gray),
                "text_regions": self._detect_text_regions(frame),
                "button_regions": self._detect_button_regions(frame),
                "app_indicators": self._detect_app_indicators(frame)
            }
            ui_states.append(ui_state)
            
        return ui_states
    
    def _detect_text_regions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in the frame."""
        # Simplified text detection - in practice, use OCR like Tesseract
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find potential text regions
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10:  # Filter small regions
                text_regions.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "area": w * h
                })
                
        return text_regions
    
    def _detect_button_regions(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect button-like regions in the frame."""
        # Simplified button detection using color and shape analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect common button colors (blue, green, red)
        button_regions = []
        
        # Blue button detection
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Green button detection
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks
        button_mask = cv2.bitwise_or(blue_mask, green_mask)
        
        # Find contours
        contours, _ = cv2.findContours(button_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 20:  # Filter small regions
                button_regions.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "area": w * h,
                    "type": "button"
                })
                
        return button_regions
    
    def _detect_app_indicators(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect app-specific indicators (notifications, modals, etc.)."""
        indicators = []
        
        # Detect notification bar
        top_region = frame[:100, :, :]
        if np.mean(top_region) > 200:  # Bright top region
            indicators.append({
                "type": "notification_bar",
                "region": "top",
                "confidence": 0.8
            })
        
        # Detect modal dialogs (dark overlay)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 100:  # Dark frame
            indicators.append({
                "type": "modal_dialog",
                "region": "full",
                "confidence": 0.7
            })
            
        return indicators
    
    def _extract_user_actions(self, frames: List[np.ndarray], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Extract user actions from frame differences."""
        actions = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Detect significant changes
            change_threshold = 30
            significant_change = np.mean(diff_gray) > change_threshold
            
            if significant_change:
                # Analyze the type of change
                action_type = self._classify_action(prev_frame, curr_frame, diff)
                
                actions.append({
                    "timestamp": timestamps[i],
                    "frame_index": i,
                    "action_type": action_type,
                    "change_magnitude": float(np.mean(diff_gray))
                })
                
        return actions
    
    def _classify_action(self, prev_frame: np.ndarray, curr_frame: np.ndarray, diff: np.ndarray) -> str:
        """Classify the type of user action based on frame differences."""
        # Simplified action classification
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Find the region with maximum change
        _, _, _, max_loc = cv2.minMaxLoc(diff_gray)
        x, y = max_loc
        
        # Analyze the change pattern
        change_region = diff_gray[max(0, y-20):min(diff_gray.shape[0], y+20),
                                 max(0, x-20):min(diff_gray.shape[1], x+20)]
        
        if np.mean(change_region) > 50:
            # Large change - likely a tap
            return "tap"
        elif np.mean(change_region) > 20:
            # Medium change - likely a scroll
            return "scroll"
        else:
            # Small change - likely a wait or minor interaction
            return "wait"
    
    def _determine_task_completion(self, ui_states: List[Dict[str, Any]], user_actions: List[Dict[str, Any]]) -> bool:
        """Determine if the task was completed based on UI states and actions."""
        if not ui_states or not user_actions:
            return False
            
        # Look for completion indicators in final UI states
        final_ui_state = ui_states[-1]
        
        # Check for success indicators
        success_indicators = [
            "success" in str(final_ui_state).lower(),
            "complete" in str(final_ui_state).lower(),
            "done" in str(final_ui_state).lower()
        ]
        
        # Check if user stopped interacting (task completion)
        if len(user_actions) > 0:
            last_action_time = user_actions[-1]["timestamp"]
            video_duration = ui_states[-1].get("frame_index", 0) / 30  # Assuming 30 fps
            
            # If user stopped interacting near the end, likely completed
            if video_duration - last_action_time < 2.0:
                return True
                
        return any(success_indicators)

class TaskPromptGenerator:
    """Generates task prompts from video analysis."""
    
    def __init__(self):
        self.llm = LLMClient()
        
    def generate_task_prompt(self, video_trace: VideoTrace) -> str:
        """Generate a natural language task prompt from video analysis."""
        log.info(f"Generating task prompt for video: {video_trace.video_path}")
        
        # Create a description of the video content
        video_description = self._create_video_description(video_trace)
        
        # Use LLM to generate task prompt
        system_prompt = """
You are an expert at analyzing user interaction videos and generating natural language task descriptions.
Given a video trace of user interactions with an Android app, generate a clear, specific task prompt that describes what the user was trying to accomplish.

Focus on:
1. The main goal the user was pursuing
2. Specific actions they took
3. The final outcome they achieved
4. Use natural, human-like language

Return only the task description, no additional commentary.
"""
        
        user_prompt = f"""
Video Analysis:
- Duration: {video_trace.timestamps[-1]:.2f} seconds
- Number of actions: {len(video_trace.user_actions)}
- UI states analyzed: {len(video_trace.ui_states)}
- Task completed: {video_trace.task_completion}

Video Description:
{video_description}

User Actions:
{json.dumps(video_trace.user_actions[:5], indent=2)}  # First 5 actions

Generate a natural language task prompt that describes what the user was trying to accomplish.
"""
        
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            task_prompt = response.choices[0].message.content.strip()
            log.info(f"Generated task prompt: {task_prompt}")
            return task_prompt
            
        except Exception as e:
            log.error(f"Error generating task prompt: {e}")
            # Fallback to basic prompt
            return f"Reproduce the user interactions shown in {video_trace.video_path.name}"
    
    def _create_video_description(self, video_trace: VideoTrace) -> str:
        """Create a description of the video content."""
        description = []
        
        # App context
        if video_trace.ui_states:
            first_state = video_trace.ui_states[0]
            if first_state.get("app_indicators"):
                description.append("App shows various UI elements including notifications and dialogs.")
        
        # Action summary
        action_types = [action["action_type"] for action in video_trace.user_actions]
        action_counts = {}
        for action_type in action_types:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
        if action_counts:
            description.append(f"User performed {len(video_trace.user_actions)} actions: {action_counts}")
        
        # UI state changes
        if len(video_trace.ui_states) > 1:
            description.append(f"UI underwent {len(video_trace.ui_states)} state changes")
        
        # Task completion
        if video_trace.task_completion:
            description.append("Task appears to have been completed successfully.")
        else:
            description.append("Task completion status unclear.")
            
        return " ".join(description)

class AITWEvaluator:
    """Main evaluator for Android in the Wild dataset."""
    
    def __init__(self):
        self.video_analyzer = AITWVideoAnalyzer()
        self.task_generator = TaskPromptGenerator()
        self.memory = EpisodicMemory()
        
    def evaluate_video(self, video_path: Path, agent_trace: AgentTrace) -> AITWScore:
        """Evaluate agent performance against a video trace."""
        log.info(f"Evaluating video: {video_path}")
        
        # Analyze the video
        video_trace = self.video_analyzer.analyze_video(video_path)
        
        # Generate task prompt
        task_prompt = self.task_generator.generate_task_prompt(video_trace)
        
        # Calculate various similarity metrics
        accuracy_score = self._calculate_accuracy_score(video_trace, agent_trace)
        robustness_score = self._calculate_robustness_score(video_trace, agent_trace)
        generalization_score = self._calculate_generalization_score(video_trace, agent_trace)
        action_similarity = self._calculate_action_similarity(video_trace, agent_trace)
        ui_state_similarity = self._calculate_ui_state_similarity(video_trace, agent_trace)
        
        return AITWScore(
            accuracy_score=accuracy_score,
            robustness_score=robustness_score,
            generalization_score=generalization_score,
            task_completion_rate=1.0 if agent_trace.task_completion else 0.0,
            average_duration=agent_trace.duration,
            action_similarity=action_similarity,
            ui_state_similarity=ui_state_similarity
        )
    
    def evaluate_episode(self, episode: Dict[str, Any], agent_trace: AgentTrace) -> AITWScore:
        """Evaluate an agent's performance against an episode trace."""
        log.info(f"Evaluating agent performance for episode: {episode.get('episode_id', 'unknown')}")
        
        # Calculate basic scores based on episode data
        accuracy_score = self._calculate_episode_accuracy_score(episode, agent_trace)
        robustness_score = self._calculate_episode_robustness_score(episode, agent_trace)
        generalization_score = self._calculate_episode_generalization_score(episode, agent_trace)
        
        # Calculate task completion rate
        task_completion_rate = 1.0 if agent_trace.task_completion else 0.0
        
        # Calculate average duration
        average_duration = agent_trace.duration
        
        # For demo purposes, use simplified similarity scores
        action_similarity = 0.5  # Placeholder
        ui_state_similarity = 0.5  # Placeholder
        
        return AITWScore(
            accuracy_score=accuracy_score,
            robustness_score=robustness_score,
            generalization_score=generalization_score,
            task_completion_rate=task_completion_rate,
            average_duration=average_duration,
            action_similarity=action_similarity,
            ui_state_similarity=ui_state_similarity
        )
    
    def _calculate_episode_accuracy_score(self, episode: Dict[str, Any], agent_trace: AgentTrace) -> float:
        """Calculate accuracy score for episode evaluation."""
        # Compare episode actions with agent actions
        episode_actions = [step.get('action', 'unknown') for step in episode.get('steps', [])]
        agent_actions = [action.get('action', 'unknown') for action in agent_trace.actions]
        
        if not episode_actions:
            return 0.0
            
        # Calculate action overlap
        episode_action_set = set(episode_actions)
        agent_action_set = set(agent_actions)
        
        overlap = len(episode_action_set.intersection(agent_action_set))
        accuracy = overlap / len(episode_action_set)
        
        return min(accuracy, 1.0)
    
    def _calculate_episode_robustness_score(self, episode: Dict[str, Any], agent_trace: AgentTrace) -> float:
        """Calculate robustness score for episode evaluation."""
        # Robustness based on error handling and task completion
        if agent_trace.task_completion:
            return 0.8  # Good robustness if task completed
        else:
            return 0.2  # Lower robustness if task failed
    
    def _calculate_episode_generalization_score(self, episode: Dict[str, Any], agent_trace: AgentTrace) -> float:
        """Calculate generalization score for episode evaluation."""
        # Generalization based on how well agent adapted to the episode
        if agent_trace.task_completion:
            return 0.7  # Good generalization if task completed
        else:
            return 0.3  # Lower generalization if task failed
    
    def _calculate_accuracy_score(self, video_trace: VideoTrace, agent_trace: AgentTrace) -> float:
        """Calculate how accurately the agent reproduced the video flow."""
        # Compare action sequences
        video_actions = [action["action_type"] for action in video_trace.user_actions]
        agent_actions = [action.get("action", "unknown") for action in agent_trace.actions]
        
        # Calculate sequence similarity
        if not video_actions or not agent_actions:
            return 0.0
            
        # Simple similarity based on action type overlap
        video_action_set = set(video_actions)
        agent_action_set = set(agent_actions)
        
        if not video_action_set:
            return 0.0
            
        overlap = len(video_action_set.intersection(agent_action_set))
        accuracy = overlap / len(video_action_set)
        
        return min(accuracy, 1.0)
    
    def _calculate_robustness_score(self, video_trace: VideoTrace, agent_trace: AgentTrace) -> float:
        """Calculate how robust the agent was in handling variations."""
        # Robustness is measured by how well the agent handled unexpected situations
        # For now, use a simplified metric based on error handling
        
        error_count = sum(1 for action in agent_trace.actions if action.get("error"))
        total_actions = len(agent_trace.actions)
        
        if total_actions == 0:
            return 0.0
            
        # Robustness decreases with more errors
        robustness = 1.0 - (error_count / total_actions)
        return max(robustness, 0.0)
    
    def _calculate_generalization_score(self, video_trace: VideoTrace, agent_trace: AgentTrace) -> float:
        """Calculate how well the agent generalizes to new scenarios."""
        # Generalization is measured by how well the agent adapts to different UI patterns
        # For now, use a simplified metric based on task completion and efficiency
        
        if not agent_trace.task_completion:
            return 0.0
            
        # Compare efficiency (actions per second)
        video_efficiency = len(video_trace.user_actions) / video_trace.timestamps[-1] if video_trace.timestamps else 0
        agent_efficiency = len(agent_trace.actions) / agent_trace.duration if agent_trace.duration > 0 else 0
        
        if video_efficiency == 0:
            return 1.0 if agent_trace.task_completion else 0.0
            
        # Efficiency ratio (closer to 1.0 is better)
        efficiency_ratio = min(agent_efficiency / video_efficiency, video_efficiency / agent_efficiency)
        
        return efficiency_ratio
    
    def _calculate_action_similarity(self, video_trace: VideoTrace, agent_trace: AgentTrace) -> float:
        """Calculate similarity between video and agent actions."""
        video_actions = [action["action_type"] for action in video_trace.user_actions]
        agent_actions = [action.get("action", "unknown") for action in agent_trace.actions]
        
        if not video_actions or not agent_actions:
            return 0.0
            
        # Map action types for comparison
        action_mapping = {
            "tap": ["tap", "click"],
            "scroll": ["scroll"],
            "wait": ["wait"],
            "type": ["type", "input"]
        }
        
        # Calculate similarity based on mapped actions
        similar_actions = 0
        total_comparisons = min(len(video_actions), len(agent_actions))
        
        for i in range(total_comparisons):
            video_action = video_actions[i]
            agent_action = agent_actions[i]
            
            # Check if actions are similar
            for video_type, agent_types in action_mapping.items():
                if video_action == video_type and agent_action in agent_types:
                    similar_actions += 1
                    break
                    
        return similar_actions / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_ui_state_similarity(self, video_trace: VideoTrace, agent_trace: AgentTrace) -> float:
        """Calculate similarity between UI states reached."""
        if not video_trace.ui_states or not agent_trace.ui_states:
            return 0.0
            
        # Compare final UI states
        final_video_state = video_trace.ui_states[-1]
        final_agent_state = agent_trace.ui_states[-1] if agent_trace.ui_states else {}
        
        # Simple similarity based on state properties
        video_properties = set(str(final_video_state).split())
        agent_properties = set(str(final_agent_state).split())
        
        if not video_properties:
            return 0.0
            
        intersection = len(video_properties.intersection(agent_properties))
        union = len(video_properties.union(agent_properties))
        
        return intersection / union if union > 0 else 0.0

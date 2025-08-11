"""
Trace recorder for capturing detailed execution traces from the multi-agent system.
Used for comparison with Android in the Wild video traces.
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from core.logging_config import get_logger
from core.message_bus import subscribe, Message

log = get_logger("TRACE-RECORDER")

@dataclass
class ActionTrace:
    """Represents a single action in the execution trace."""
    timestamp: float
    action_type: str
    action_data: Dict[str, Any]
    ui_state_before: Optional[str]
    ui_state_after: Optional[str]
    success: bool
    error: Optional[str]
    duration: float

@dataclass
class EpisodeTrace:
    """Represents a complete episode execution trace."""
    episode_id: str
    user_goal: str
    start_time: float
    end_time: float
    actions: List[ActionTrace]
    final_ui_state: Optional[str]
    task_completed: bool
    completion_reason: str

class TraceRecorder:
    """Records detailed execution traces for comparison with video traces."""
    
    def __init__(self, output_dir: str = "logs/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_episode: Optional[EpisodeTrace] = None
        self.episode_start_time: Optional[float] = None
        
        # Subscribe to relevant message channels
        subscribe("plan", self.on_plan)
        subscribe("exec-report", self.on_exec_report)
        subscribe("episode_done", self.on_episode_done)
        
    def on_plan(self, msg: Message):
        """Handle planning messages to start episode tracking."""
        if not self.current_episode:
            # Start new episode
            self.episode_start_time = time.time()
            self.current_episode = EpisodeTrace(
                episode_id=msg.payload.get("episode_id", "unknown"),
                user_goal="",  # Will be updated when available
                start_time=self.episode_start_time,
                end_time=0.0,
                actions=[],
                final_ui_state=None,
                task_completed=False,
                completion_reason=""
            )
            log.info(f"Started recording trace for episode: {self.current_episode.episode_id}")
    
    def on_exec_report(self, msg: Message):
        """Handle execution reports to record action traces."""
        if not self.current_episode:
            return
            
        report = msg.payload["report"]
        step = report.get("step", {})
        
        # Create action trace
        action_trace = ActionTrace(
            timestamp=time.time(),
            action_type=step.get("action", "unknown"),
            action_data=step,
            ui_state_before=msg.payload.get("ui_snapshot", ""),
            ui_state_after="",  # Will be updated in next report
            success=report.get("success", False),
            error=report.get("error"),
            duration=0.0  # Will be calculated
        )
        
        # Update previous action's UI state after
        if self.current_episode.actions:
            self.current_episode.actions[-1].ui_state_after = action_trace.ui_state_before
            # Calculate duration
            self.current_episode.actions[-1].duration = (
                action_trace.timestamp - self.current_episode.actions[-1].timestamp
            )
        
        self.current_episode.actions.append(action_trace)
        
        log.debug(f"Recorded action: {action_trace.action_type}")
    
    def on_episode_done(self, msg: Message):
        """Handle episode completion to finalize trace recording."""
        if not self.current_episode:
            return
            
        # Update episode completion info
        self.current_episode.end_time = time.time()
        self.current_episode.task_completed = True
        self.current_episode.completion_reason = msg.payload.get("reason", "unknown")
        
        # Update final UI state if available
        if self.current_episode.actions:
            self.current_episode.final_ui_state = self.current_episode.actions[-1].ui_state_after
        
        # Calculate duration for last action
        if self.current_episode.actions:
            last_action = self.current_episode.actions[-1]
            last_action.duration = self.current_episode.end_time - last_action.timestamp
        
        # Save the trace
        self.save_trace(self.current_episode)
        
        log.info(f"Completed trace recording for episode: {self.current_episode.episode_id}")
        self.current_episode = None
        self.episode_start_time = None
    
    def save_trace(self, episode_trace: EpisodeTrace):
        """Save episode trace to file."""
        trace_file = self.output_dir / f"episode_{episode_trace.episode_id}.json"
        
        # Convert to dict for JSON serialization
        trace_data = asdict(episode_trace)
        
        # Add metadata
        trace_data["metadata"] = {
            "recorder_version": "1.0",
            "recording_timestamp": time.time(),
            "total_actions": len(episode_trace.actions),
            "episode_duration": episode_trace.end_time - episode_trace.start_time
        }
        
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
            
        log.info(f"Saved trace to: {trace_file}")
    
    def get_latest_trace(self) -> Optional[EpisodeTrace]:
        """Get the most recently recorded trace."""
        return self.current_episode
    
    def load_trace(self, episode_id: str) -> Optional[EpisodeTrace]:
        """Load a trace from file."""
        trace_file = self.output_dir / f"episode_{episode_id}.json"
        
        if not trace_file.exists():
            return None
            
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            # Reconstruct EpisodeTrace from dict
            actions = []
            for action_data in trace_data["actions"]:
                action = ActionTrace(
                    timestamp=action_data["timestamp"],
                    action_type=action_data["action_type"],
                    action_data=action_data["action_data"],
                    ui_state_before=action_data["ui_state_before"],
                    ui_state_after=action_data["ui_state_after"],
                    success=action_data["success"],
                    error=action_data.get("error"),
                    duration=action_data["duration"]
                )
                actions.append(action)
            
            episode = EpisodeTrace(
                episode_id=trace_data["episode_id"],
                user_goal=trace_data["user_goal"],
                start_time=trace_data["start_time"],
                end_time=trace_data["end_time"],
                actions=actions,
                final_ui_state=trace_data["final_ui_state"],
                task_completed=trace_data["task_completed"],
                completion_reason=trace_data["completion_reason"]
            )
            
            return episode
            
        except Exception as e:
            log.error(f"Error loading trace {episode_id}: {e}")
            return None
    
    def list_traces(self) -> List[str]:
        """List all available trace files."""
        trace_files = list(self.output_dir.glob("episode_*.json"))
        return [f.stem.replace("episode_", "") for f in trace_files]

class TraceAnalyzer:
    """Analyzes execution traces for insights and comparison."""
    
    @staticmethod
    def analyze_trace(episode_trace: EpisodeTrace) -> Dict[str, Any]:
        """Analyze an episode trace for insights."""
        if not episode_trace.actions:
            return {"error": "No actions in trace"}
        
        # Basic statistics
        total_actions = len(episode_trace.actions)
        successful_actions = sum(1 for action in episode_trace.actions if action.success)
        failed_actions = total_actions - successful_actions
        
        # Action type distribution
        action_types = {}
        for action in episode_trace.actions:
            action_type = action.action_type
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        # Timing analysis
        total_duration = episode_trace.end_time - episode_trace.start_time
        avg_action_duration = sum(action.duration for action in episode_trace.actions) / total_actions
        
        # Error analysis
        errors = [action.error for action in episode_trace.actions if action.error]
        
        return {
            "episode_id": episode_trace.episode_id,
            "user_goal": episode_trace.user_goal,
            "task_completed": episode_trace.task_completed,
            "completion_reason": episode_trace.completion_reason,
            "total_duration": total_duration,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "action_types": action_types,
            "avg_action_duration": avg_action_duration,
            "errors": errors,
            "error_count": len(errors)
        }
    
    @staticmethod
    def compare_traces(trace1: EpisodeTrace, trace2: EpisodeTrace) -> Dict[str, Any]:
        """Compare two execution traces."""
        analysis1 = TraceAnalyzer.analyze_trace(trace1)
        analysis2 = TraceAnalyzer.analyze_trace(trace2)
        
        # Calculate similarity metrics
        action_type_overlap = set(analysis1["action_types"].keys()).intersection(
            set(analysis2["action_types"].keys())
        )
        
        total_action_types = set(analysis1["action_types"].keys()).union(
            set(analysis2["action_types"].keys())
        )
        
        action_type_similarity = len(action_type_overlap) / len(total_action_types) if total_action_types else 0.0
        
        # Duration similarity
        duration_ratio = min(analysis1["total_duration"], analysis2["total_duration"]) / max(
            analysis1["total_duration"], analysis2["total_duration"]
        ) if max(analysis1["total_duration"], analysis2["total_duration"]) > 0 else 0.0
        
        return {
            "trace1_analysis": analysis1,
            "trace2_analysis": analysis2,
            "similarity_metrics": {
                "action_type_similarity": action_type_similarity,
                "duration_similarity": duration_ratio,
                "success_rate_similarity": 1.0 - abs(
                    analysis1["success_rate"] - analysis2["success_rate"]
                ),
                "task_completion_similarity": 1.0 if (
                    analysis1["task_completed"] == analysis2["task_completed"]
                ) else 0.0
            }
        }

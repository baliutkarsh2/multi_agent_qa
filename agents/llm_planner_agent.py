"""LLM-powered planner agent."""
from __future__ import annotations
import uuid
from typing import Dict, Any, List
from core.message_bus import Message, publish, subscribe
from core.registry import register_agent
from core.memory import EpisodicMemory
from core.episode import EpisodeContext
from core.llm_client import LLMClient
from env.android_interface import UIState
import os
from core.logging_config import get_logger

log = get_logger("LLM-PLANNER")

@register_agent("llm_planner")
class LLMPlannerAgent:
    def __init__(self):
        self.llm = LLMClient()
        self.memory = EpisodicMemory()
        subscribe("exec-report", self.on_exec_report)

    def on_exec_report(self, msg: Message):
        # After a step is executed, plan the next one.
        episode_id = msg.payload["episode_id"]
        ui_state = UIState(msg.payload["ui_snapshot"])
        
        history = self.memory.retrieve(episode_id) or []
        
        # Failsafe for empty history
        if not history:
            log.error(f"History not found for episode {episode_id}. Ending episode.")
            publish(Message("LLM-PLANNER", "episode_done", {"reason": "History lost."}))
            return

        user_goal = history[0].get("user_goal", "No goal found in history.")
        
        self.act(user_goal, ui_state, EpisodeContext(id=episode_id, user_goal=user_goal))

    def act(self, user_goal: str, ui_state: UIState, episode: EpisodeContext):
        history = self.memory.retrieve(episode.id) or []
        
        if not history:
            history.append({"user_goal": user_goal})
            
        log.info(f"Planning next action for goal: {user_goal}")
        
        action = self.llm.request_next_action(user_goal, ui_state.xml, history)
        
        if not action or "action" not in action:
            log.warning("LLM did not return a valid action. Ending episode.")
            publish(Message("LLM-PLANNER", "episode_done", {"reason": "No further actions from LLM."}))
            return
            
        history.append(action)
        self.memory.store(episode.id, history, tags=["history"])
        
        publish(Message("LLM-PLANNER", "plan", {"step": action, "episode_id": episode.id})) 
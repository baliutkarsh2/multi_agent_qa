"""LLM-powered supervisor agent."""
from __future__ import annotations
import uuid
from collections import defaultdict
from typing import Dict, Any, List
from core.message_bus import subscribe, Message, publish
from core.registry import register_agent
from evaluation.evaluator import EpisodeEvaluator
from core.memory import NarrativeMemory
from core.logging_config import get_logger

log=get_logger("LLM-SUPERVISOR")

@register_agent("llm_supervisor")
class LLMSupervisorAgent:
    def __init__(self):
        self._eps={}
        self.eval=EpisodeEvaluator()
        subscribe("exec-report",self.on_exec)
        subscribe("verify-report",self.on_verify)

    def on_exec(self,msg:Message):
        ep=self._eps.setdefault(msg.payload["episode_id"],{"exec":[],"verify":[]})
        ep["exec"].append(msg.payload)

    def on_verify(self,msg:Message):
        ep=self._eps[msg.payload["episode_id"]]
        ep["verify"].append(msg.payload)
        need=sum(1 for r in ep["exec"] if r["report"]["step"]["action"]=="verify")
        
        # If all verify steps are complete, finish the episode.
        if len(ep["verify"])>=need:
            score=self.eval.evaluate(ep["exec"],ep["verify"])
            NarrativeMemory().store(f"ep-{msg.payload['episode_id']}",score.model_dump(),tags=["ep"])
            log.info(f"Episode {msg.payload['episode_id']} summary: {score.model_dump_json(indent=2)}")
            
            # Signal that the episode is done
            publish(Message("LLM-SUPERVISOR", "episode_done", {"reason": "All steps verified."})) 
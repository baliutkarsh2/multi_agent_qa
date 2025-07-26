"""LLM-powered verifier agent."""
from __future__ import annotations
import uuid
from typing import Dict, Any, List
from core.message_bus import subscribe, publish, Message
from core.registry import register_agent
from core.memory import EpisodicMemory
from core.llm_client import LLMClient
from core.logging_config import get_logger
from env.android_interface import AndroidDevice

log = get_logger("LLM-VERIFIER")

@register_agent("llm_verifier")
class LLMVerifierAgent:
    def __init__(self, device: AndroidDevice):
        self.device = device
        self.llm = LLMClient()
        subscribe("exec-report", self.on_exec)

    def on_exec(self, msg: Message):
        step=msg.payload["report"]["step"]
        eid=msg.payload["episode_id"]
        if step["action"]!="verify": return
        ui_xml=self.device.get_ui_tree().xml
        log.info(f"Verifying step: {step}")
        result=self.llm.request_next_action(f"Verify: {step.get('text','')}",ui_xml,[])
        verified=result.get("verified",False)
        publish(Message("LLM-VERIFIER","verify-report",{"episode_id":eid,"step_id":step["step_id"],"verified":verified})) 
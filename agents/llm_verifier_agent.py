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
        step = msg.payload["report"]["step"]
        eid = msg.payload["episode_id"]
        
        # Only verify steps that are marked as verification actions
        if step["action"] != "verify":
            return
            
        log.info(f"Verifying step: {step}")
        
        # Get current UI state
        ui_xml = self.device.get_ui_tree().xml
        
        # Create a description of what we're verifying
        action_description = self._create_verification_description(step)
        
        # Use the specialized verification method
        result = self.llm.verify_action(action_description, ui_xml)
        
        # Extract verification result
        verified = result.get("verified", False)
        reason = result.get("reason", "No reason provided")
        confidence = result.get("confidence", 0.5)
        
        log.info(f"Verification result: {verified} (confidence: {confidence}) - {reason}")
        
        # Publish verification report
        publish(Message(
            "LLM-VERIFIER", 
            "verify-report", 
            {
                "episode_id": eid,
                "step_id": step["step_id"],
                "verified": verified,
                "reason": reason,
                "confidence": confidence,
                "ui_xml": ui_xml  # Include UI state for debugging
            }
        ))

    def _create_verification_description(self, step: Dict[str, Any]) -> str:
        """Create a human-readable description of what we're verifying."""
        action = step.get("action", "")
        resource_id = step.get("resource_id", "")
        text = step.get("text", "")
        rationale = step.get("rationale", "")
        
        description = f"Action: {action}"
        if resource_id:
            description += f", Resource ID: {resource_id}"
        if text:
            description += f", Text: {text}"
        if rationale:
            description += f", Rationale: {rationale}"
            
        return description 
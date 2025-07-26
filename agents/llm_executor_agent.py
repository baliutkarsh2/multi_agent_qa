"""LLM-powered executor agent."""
from __future__ import annotations
import time, uuid
from typing import Dict, Any, List
from core.message_bus import subscribe, publish, Message
from core.registry import register_agent, get_agent
from core.memory import EpisodicMemory
from core.llm_client import LLMClient
from core.logging_config import get_logger
from env.android_interface import AndroidDevice, UIState
from env.gesture_utils import tap_at, scroll
from env.ui_utils import get_nth_by_res_id, get_nth_by_text, find_all_by_res_id_and_text, select_nth

log = get_logger("LLM-EXECUTOR")

@register_agent("llm_executor")
class LLMExecutorAgent:
    def __init__(self, device: AndroidDevice):
        self.device = device
        self.memory = EpisodicMemory()
        subscribe("plan", self.on_plan)

    def on_plan(self, msg: Message):
        step = msg.payload["step"]
        eid  = msg.payload["episode_id"]
        log.info(f"Executing step: {step}")
        result={"step":step,"success":False,"error":None}
        
        try:
            # All actions that need the current UI state
            ui_xml = self.device.get_ui_tree().xml
            act=step["action"]

            if act=="launch_app":
                self.device.launch_app(step["package"])
            elif act=="tap":
                coord=None
                order = step.get("order", 1)

                # Prioritize searching by both resource-id and text for max accuracy
                if "resource_id" in step and "text" in step:
                    matches = find_all_by_res_id_and_text(ui_xml, step["resource_id"], step["text"])
                    coord = select_nth(matches, order)

                # Fallback to resource-id only
                if not coord and "resource_id" in step:
                    coord=get_nth_by_res_id(ui_xml,step["resource_id"], order)
                
                # Fallback to text only
                if not coord and step.get("text"):
                    coord=get_nth_by_text(ui_xml,step["text"], order)
                
                if not coord: raise RuntimeError("Element not found")
                tap_at(self.device,coord)
            elif act=="press_key":
                self.device.press_key(step["key"])
            elif act=="verify":
                post_xml=self.device.get_ui_tree().xml
                result["success"]=bool(
                    ("resource_id" in step and get_nth_by_res_id(post_xml,step["resource_id"],1)) or
                    ("text" in step and get_nth_by_text(post_xml,step["text"],1))
                )
            elif act=="scroll":
                scroll(self.device, step["direction"])
            elif act=="wait":
                import time; time.sleep(step["duration"])
            
            result["success"] = True if act not in ["verify"] else result.get("success", False)
        
        except Exception as e:
            result["error"]=str(e)
            log.error(f"Execution error: {e}")
        
        shot=self.device.screenshot(step.get("step_id", "action"))
        
        # Store the updated history
        history = self.memory.retrieve(eid) or []
        history.append(result)
        self.memory.store(eid, history, tags=["history"])
        
        # Publish the report, which the planner will listen for
        publish(Message(
            "LLM-EXECUTOR",
            "exec-report",
            {"report":result,"episode_id":eid, "ui_snapshot": self.device.get_ui_tree().xml}
        )) 
"""
LLM client wrapper for OpenAI API with structured outputs and function calling.
"""
from __future__ import annotations
import json
from openai import OpenAI
from typing import Any, Dict
from core.config import OPENAI_API_KEY
from core.logging_config import get_logger

log = get_logger("LLM-CLIENT")

class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def request_next_action(self, goal: str, ui_xml: str, history: list[Dict[str,Any]]) -> Dict[str,Any]:
        system = """
You are a mobile UI automation planner. Your goal is to create a precise and correct action to achieve a user's goal.

**RULES FOR TAPPING ACCURATELY:**
1.  **Use Both `resource-id` and `text`:** For maximum accuracy, you **MUST** provide both `resource-id` and `text` in your `tap` action whenever both are available in the UI XML. This is the best way to ensure the correct element is selected.
2.  **Handle Duplicates with `order`:** If multiple elements on the screen have the same identifiers, you **MUST** use the `order` field (e.g., `order: 2` for the second match).
3.  **Fallback to Single Identifiers:** Only use a single identifier (`resource-id` or `text`) if providing both is not possible.

Available actions and schemas:
- launch_app: {step_id,action:"launch_app",package,rationale}
- tap:        {step_id,action:"tap",resource_id?,text?,order?,rationale}
- press_key:  {step_id,action:"press_key",key:"home"|"back"|"recents"|"enter",rationale}
- verify:     {step_id,action:"verify",resource_id?,text?,rationale}
- scroll:     {step_id,action:"scroll",direction:"up|down",until_resource_id?,until_text?,rationale}
- wait:       {step_id,action:"wait",duration (secs),rationale}

**IMPORTANT OPERATIONAL RULES**:
1. To submit a search or form, you **MUST** use `press_key` with the `enter` key.

Reply with exactly one JSON object matching one of the schemas. Do not add extra commentary or keys.
"""
        user = f"""
Goal: {goal}

Current UI XML:
{ui_xml}

History:
{json.dumps(history, indent=2)}

Provide the *next* action to take as a single JSON object.
"""
        log.debug("Requesting next action from LLM")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.0,
            max_tokens=256
        )
        content = resp.choices[0].message.content
        log.debug(f"LLM response: {content}")
        return json.loads(content) 
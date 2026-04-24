"""
JSON parser utility for extracting structured data from LLM responses.
"""

import json
import re


def parse_json_object(text: str) -> dict:
    """
    Extract a JSON object from LLM response text.

    Handles common LLM output patterns:
    - Raw JSON
    - JSON wrapped in ```json ... ``` code fences
    - JSON embedded in surrounding prose
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fences
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try finding the first { ... } block
    brace_pattern = r"\{.*\}"
    match = re.search(brace_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return the raw text wrapped in a dict
    return {"raw_output": text}

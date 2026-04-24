"""
Scheduler Agent — optimizes posting times and creates a weekly schedule.

Pure text generation agent (no tool calling needed).
"""

import json

from utils.parser import parse_json_object
from utils.prompts import SCHEDULER_SYSTEM


def run_scheduler_agent(
    client,
    content_plan: str,
    audience_summary: str,
    platform: str,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Run the Scheduler agent.

    Args:
        client: OpenAI-compatible client (pointed at Groq).
        content_plan: JSON string from the Content Planner agent.
        audience_summary: JSON string from the Audience Research agent.
        platform: Target social media platform.

    Returns:
        Parsed schedule as a dict.
    """
    user_message = (
        f"Create an optimized posting schedule for {platform}.\n\n"
        f"**Content Plan:**\n{content_plan}\n\n"
        f"**Audience Behavior Data:**\n{audience_summary}\n\n"
        f"Optimize the schedule based on when the target audience is most "
        f"active on {platform} and the types of content being posted."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SCHEDULER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
    )

    text = response.choices[0].message.content or ""
    return parse_json_object(text)

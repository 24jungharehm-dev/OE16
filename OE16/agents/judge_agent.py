"""
Judge Agent — evaluates the complete strategy package and decides
whether to approve it or send it back for revision.

Pure text generation agent (no tool calling needed).
"""

import json

from utils.parser import parse_json_object
from utils.prompts import JUDGE_EVAL_PROMPT, JUDGE_SYSTEM


def run_judge_agent(
    client,
    strategy: str,
    audience: str,
    content: str,
    schedule: str,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Run the Judge agent to evaluate the entire strategy package.

    Args:
        client: OpenAI-compatible client (pointed at Groq).
        strategy: JSON string of the strategy output.
        audience: JSON string of the audience research output.
        content: JSON string of the content plan output.
        schedule: JSON string of the schedule output.

    Returns:
        Parsed evaluation as a dict with score, verdict, and feedback.
    """
    eval_prompt = JUDGE_EVAL_PROMPT.format(
        strategy=strategy,
        audience=audience,
        content=content,
        schedule=schedule,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": eval_prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    return parse_json_object(text)

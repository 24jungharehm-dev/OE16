"""
Audience Research Agent — identifies and profiles target audiences.

Uses Tavily search via tool calling to gather real demographic and
behavioral data about the target audience on the specified platform.
"""

import json

from tools.tavily_tool import TAVILY_TOOL_DEFINITION, tavily_search_tool
from utils.parser import parse_json_object
from utils.prompts import AUDIENCE_RESEARCH_SYSTEM


def run_audience_research_agent(
    client,
    strategy_summary: str,
    platform: str,
    tavily_api_key: str,
    max_turns: int = 10,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Run the Audience Research agent.

    Args:
        client: OpenAI-compatible client (pointed at Groq).
        strategy_summary: JSON string or summary from the Strategy Planner.
        platform: Target social media platform.
        tavily_api_key: API key for Tavily searches.
        max_turns: Maximum tool-calling loop iterations.

    Returns:
        Parsed audience research as a dict.
    """
    user_message = (
        f"Research the target audience for this social media strategy:\n\n"
        f"**Strategy:**\n{strategy_summary}\n\n"
        f"**Platform:** {platform}\n\n"
        f"Use the search tool to find real data about audience demographics "
        f"and behaviors on {platform} for this niche."
    )

    messages = [
        {"role": "system", "content": AUDIENCE_RESEARCH_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    tools = [TAVILY_TOOL_DEFINITION]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name == "tavily_search":
                    result = tavily_search_tool(
                        query=fn_args["query"],
                        api_key=tavily_api_key,
                    )
                else:
                    result = f"Unknown tool: {fn_name}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            text = choice.message.content or ""
            return parse_json_object(text)

    return {"error": "Audience research agent exceeded maximum tool-calling turns."}

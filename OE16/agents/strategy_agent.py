"""
Strategy Planner Agent — researches trends and creates high-level strategy.

Uses Tavily search via tool calling to gather real-time market intelligence
before formulating the strategy.
"""

import json

from tools.tavily_tool import TAVILY_TOOL_DEFINITION, tavily_search_tool
from utils.parser import parse_json_object
from utils.prompts import STRATEGY_PLANNER_SYSTEM


def run_strategy_planner_agent(
    client,
    product: str,
    platform: str,
    tavily_api_key: str,
    feedback: str | None = None,
    max_turns: int = 10,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Run the Strategy Planner agent.

    Args:
        client: OpenAI-compatible client (pointed at Groq).
        product: Brand or product description.
        platform: Target social media platform.
        tavily_api_key: API key for Tavily searches.
        feedback: Optional feedback from the Judge agent for refinement.
        max_turns: Maximum tool-calling loop iterations.

    Returns:
        Parsed strategy as a dict.
    """
    user_message = (
        f"Create a social media strategy for:\n"
        f"- Product/Brand: {product}\n"
        f"- Platform: {platform}\n"
    )

    if feedback:
        user_message += (
            f"\n⚠️ IMPORTANT — Previous strategy was rejected by the evaluator. "
            f"Here is their feedback:\n{feedback}\n\n"
            f"Incorporate this feedback to improve the strategy."
        )

    messages = [
        {"role": "system", "content": STRATEGY_PLANNER_SYSTEM},
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

        # If the model wants to call a tool
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append the assistant message (with tool calls) to history
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
            # Model has produced its final answer
            text = choice.message.content or ""
            return parse_json_object(text)

    # Fallback if max turns exceeded
    return {"error": "Strategy planner exceeded maximum tool-calling turns."}

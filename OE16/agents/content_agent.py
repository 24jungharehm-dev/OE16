"""
Content Planner Agent — designs a 7-day content calendar.

Uses the product catalog tool to get product-specific details
for creating targeted content ideas.
"""

import json

from tools.catalog_tool import CATALOG_TOOL_DEFINITION, product_catalog_tool
from utils.parser import parse_json_object
from utils.prompts import CONTENT_PLANNER_SYSTEM


def run_content_planner_agent(
    client,
    strategy_summary: str,
    audience_summary: str,
    platform: str,
    product: str,
    max_turns: int = 10,
    model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Run the Content Planner agent.

    Args:
        client: OpenAI-compatible client (pointed at Groq).
        strategy_summary: JSON string from the Strategy Planner.
        audience_summary: JSON string from the Audience Research agent.
        platform: Target social media platform.
        product: Product/brand category for catalog lookup.
        max_turns: Maximum tool-calling loop iterations.

    Returns:
        Parsed content plan as a dict.
    """
    user_message = (
        f"Create a 7-day content calendar for {platform}.\n\n"
        f"**Strategy:**\n{strategy_summary}\n\n"
        f"**Target Audience:**\n{audience_summary}\n\n"
        f"**Product Category:** {product}\n\n"
        f"Use the product_catalog tool to get specific product details "
        f"for creating targeted content."
    )

    messages = [
        {"role": "system", "content": CONTENT_PLANNER_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    tools = [CATALOG_TOOL_DEFINITION]

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

                if fn_name == "product_catalog":
                    result = product_catalog_tool(
                        category=fn_args.get("category", product),
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

    return {"error": "Content planner agent exceeded maximum tool-calling turns."}

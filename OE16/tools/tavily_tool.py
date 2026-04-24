"""
Tavily search tool for real-time web research.

Provides both a callable function and an OpenAI-compatible tool definition
for use with function/tool calling.
"""

from tavily import TavilyClient


# OpenAI-compatible tool definition for agent tool calling
TAVILY_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": (
            "Search the web for current information about trends, competitors, "
            "audience demographics, social media strategies, and market data. "
            "Use this to ground your analysis in real, up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information.",
                }
            },
            "required": ["query"],
        },
    },
}


def tavily_search_tool(query: str, api_key: str) -> str:
    """
    Execute a Tavily search and return formatted results.

    Args:
        query: The search query string.
        api_key: Tavily API key.

    Returns:
        Formatted string of search results with titles, URLs, and content.
    """
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = response.get("results", [])
        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            formatted.append(
                f"**Result {i}: {title}**\n"
                f"URL: {url}\n"
                f"Content: {content}\n"
            )

        return "\n---\n".join(formatted)

    except Exception as e:
        return f"Search error for '{query}': {str(e)}"

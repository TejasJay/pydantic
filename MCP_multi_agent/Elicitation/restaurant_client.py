import asyncio
from typing import Any

from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

import json
from pydantic_ai.agent import AgentRunResult


# --------------------------
# Helper Function for JSON
# --------------------------
def agent_result_to_json(result) -> str:
    """Convert AgentRunResult into a clean JSON string."""
    data = {
        "final_output": result.output,
        "usage": vars(result.usage) if getattr(result, "usage", None) else None,
        "messages": [],
    }
    # Extract structured message history (if available)
    for msg in getattr(result._state, "message_history", []):
        parts = []
        for part in getattr(msg, "parts", []):
            parts.append({
                "type": part.__class__.__name__,
                "content": getattr(part, "content", None),
                "tool_name": getattr(part, "tool_name", None),
                "args": getattr(part, "args", None),
            })
        data["messages"].append({
            "role": msg.__class__.__name__,
            "timestamp": getattr(msg, "timestamp", None).isoformat() if getattr(msg, "timestamp", None) else None,
            "parts": parts,
        })
    print("\nFinal Output:")
    print(data["final_output"])
    print('\n')

    return json.dumps(data, 
                      indent=2, 
                      ensure_ascii=False)



# --------------------------
# Elicitation handler
# --------------------------
async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f"\n{params.message}")

    if not params.requestedSchema:
        response = input("Response: ")
        return ElicitResult(action="accept", content={"response": response})

    properties = params.requestedSchema["properties"]
    data = {}
    for field, info in properties.items():
        description = info.get("description", field)

        if field == "party_size":
            while True:
                try:
                    value_int = int(input(f"{description}: "))
                    if not (1 <= value_int <= 8):
                        print("❌ Party size must be between 1 and 8.")
                        continue
                    data[field] = value_int
                    break
                except ValueError:
                    print("❌ Please enter a valid integer.")
        elif info.get("type") == "integer":
            data[field] = int(input(f"{description}: "))
        else:
            data[field] = input(f"{description}: ")

    # Confirm
    confirm = input("\nConfirm booking? (y/n/c): ").lower()
    if confirm == "y":
        print("Booking details:", data)
        return ElicitResult(action="accept", content=data)
    elif confirm == "n":
        return ElicitResult(action="decline")
    else:
        return ElicitResult(action="cancel")


# --------------------------
# Setup MCP Server connection
# --------------------------
restaurant_server = MCPServerStdio(
    "python3",
    args=["MCP_multi_agent/Elicitation/restaurant_server.py"],
    elicitation_callback=handle_elicitation,
)


# --------------------------
# Create agent
# --------------------------
agent = Agent(
    "google-gla:gemini-1.5-pro",
    toolsets=[restaurant_server],
    system_prompt=(
        "You are a booking assistant. "
        "You must ALWAYS call the 'book_table' tool to handle reservations. "
        "Once the tool returns, you MUST return its result to the user exactly as provided, "
        "without adding or changing anything."
    ),
)


# --------------------------
# Runner
# --------------------------
async def main():
    async with agent:
        result = await agent.run("Book me a table")
        print("\n--- JSON Result ---")
        print(agent_result_to_json(result))


if __name__ == "__main__":
    asyncio.run(main())

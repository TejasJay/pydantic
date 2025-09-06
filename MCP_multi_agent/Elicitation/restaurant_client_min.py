import asyncio
from typing import Any
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic_ai.mcp import MCPServerStdio

# --------------------------
# Elicitation callback
# --------------------------

async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    print("\n🔥 handle_elicitation CALLED!")
    print(params.message)

    if not params.requestedSchema:
        response = input("Response: ")
        return ElicitResult(action="accept", content={"response": response})

    data = {}
    for field, info in params.requestedSchema["properties"].items():
        desc = info.get("description", field)
        value = input(f"{desc}: ")
        if info.get("type") == "integer":
            value = int(value)
        data[field] = value

    confirm = input("\nConfirm booking? (y/n/c): ").lower()
    if confirm == "y":
        return ElicitResult(action="accept", content=data)
    elif confirm == "n":
        return ElicitResult(action="decline")
    else:
        return ElicitResult(action="cancel")

# --------------------------
# Minimal test runner
# --------------------------

async def main():
    restaurant_server = MCPServerStdio(
        "python3",
        args=["MCP_multi_agent/Elicitation/restaurant_server_min.py"],
        elicitation_callback=handle_elicitation
    )

    async with restaurant_server as session:
        print("\n✅ Connected to restaurant server")

        # Correct call for your API (name, args dict)
        result = await session.call_tool("book_table", {})
        print("\n--- Tool Result ---")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

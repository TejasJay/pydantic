
import asyncio
from typing import Any

from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f'\n{params.message}')

    if not params.requestedSchema:
        print(params.requestedSchema)
        response = input('Response: ')
        return ElicitResult(action='accept', content={'response': response})

    # Collect data for each field
    properties = params.requestedSchema['properties']
    data = {}

    for field, info in properties.items():
        description = info.get('description', field)

        value = input(f'{description}: ')

        # Convert to proper type based on JSON schema
        if info.get('type') == 'integer':
            data[field] = int(value)
        else:
            data[field] = value

    # Confirm
    confirm = input('\nConfirm booking? (y/n/c): ').lower()

    if confirm == 'y':
        print('Booking details:', data)
        return ElicitResult(action='accept', content=data)
    elif confirm == 'n':
        return ElicitResult(action='decline')
    else:
        return ElicitResult(action='cancel')


# Set up MCP server connection
restaurant_server = MCPServerStdio(
    'python3', args=["MCP_multi_agent/Elicitation/restaurant_server.py"], 
    elicitation_callback=handle_elicitation,

)

# Create agent
agent = Agent('google-gla:gemini-1.5-pro', 
              toolsets=[restaurant_server],
            system_prompt="You are a booking assistant. You must ALWAYS call the 'book_table' tool to handle reservations. Never answer directly."
)

async def main():
    """Run the agent to book a restaurant table."""
    async with agent:
        result = await agent.run('Book me a table')
        print(f'\nResult: {result.output}')


if __name__ == '__main__':
    asyncio.run(main())
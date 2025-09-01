from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
import asyncio

server = MCPServerStreamableHTTP(url = 'http://localhost:8000/mcp')  
agent = Agent('google-gla:gemini-1.5-pro', toolsets=[server])  

async def main():
    async with agent:  
        result = await agent.run('what is 456748 added to 678934?')
    print(result.output)
    #> There are 9,208 days between January 1, 2000, and March 18, 2025.

asyncio.run(main())
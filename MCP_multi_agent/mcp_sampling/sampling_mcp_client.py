from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

server = MCPServerStdio(command='python3', args=['MCP_multi_agent/generate_svg.py'])

# 1. Use a general-purpose model that supports function calling for the main agent
agent = Agent('google-gla:gemini-2.5-pro', toolsets=[server])


async def main():
    async with agent:
        # 2. Pass the specialized image model to set_mcp_sampling_model().
        # This tells the tool which model to use for its internal LLM call.
        agent.set_mcp_sampling_model('google-gla:gemini-2.5-flash-image-preview')
        
        result = await agent.run('Create an image of a bee riding a bike and Make it cartoonic.')
    print(result.output)

asyncio.run(main())
import re
from pathlib import Path
import sys

from mcp import SamplingMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent

app = FastMCP()


@app.tool(description="Generates an SVG image based on a subject and style. Best for non-realistic styles like 'cartoon', 'line art', 'blueprint', or 'iconic'.")
async def image_generator(ctx: Context, subject: str, style: str) -> str:
    print(f"--- Tool 'image_generator' called with subject='{subject}', style='{style}' ---", file=sys.stderr)
    prompt = f'{subject=} {style=}'
    strict_system_prompt = (
        "You are an SVG generation engine. Your sole purpose is to return raw, valid SVG code. "
        "Do NOT provide any explanation, commentary, or markdown formatting. "
        "Do not say 'Here is the SVG'. "
        "Your response must start directly with the '<svg' tag and end with '</svg>'."
    )

    result = await ctx.session.create_message(
        [SamplingMessage(role='user', content=TextContent(type='text', text=prompt))],
        max_tokens=4096,  # Increased tokens for more complex images
        system_prompt=strict_system_prompt,
    )
    assert isinstance(result.content, TextContent)

    svg_match = re.search(r'(<svg.*?</svg>)', result.content.text, re.DOTALL)

    if svg_match:
        svg_content = svg_match.group(1).strip()
        # Sanitize filename
        safe_subject = re.sub(r'[^a-zA-Z0-9_]', '', subject.replace(" ", "_"))
        safe_style = re.sub(r'[^a-zA-Z0-9_]', '', style.replace(" ", "_"))
        path = Path(f'{safe_subject}_{safe_style}.svg')

        path.write_text(svg_content)
        print(f"--- Successfully wrote {len(svg_content)} bytes to {path} ---", file=sys.stderr)
        return f'Image file written to {path}.'
    else:
        print(f"--- FAILED: No SVG content found in model output. ---", file=sys.stderr)
        return "I was unable to generate the SVG image because the model did not return valid code."


if __name__ == '__main__':
    app.run()
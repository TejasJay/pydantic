from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(name="Restaurant Booking")

class BookingDetails(BaseModel):
    restaurant: str = Field(description="Name of the restaurant")
    party_size: int = Field(description="Number of people", ge=1, le=8)
    date: str = Field(description="Reservation date (DD-MM-YYYY)")

@mcp.tool()
async def book_table(ctx: Context) -> dict:
    """Book a restaurant table with user input."""
    result = await ctx.elicit(
        message="Please provide your booking details:",
        schema=BookingDetails
    )

    if result.action == "accept" and result.data:
        return {"status": "success", "booking": result.data}
    elif result.action == "decline":
        return {"status": "declined"}
    else:
        return {"status": "cancelled"}

if __name__ == "__main__":
    mcp.run(transport="stdio")

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, ValidationError

mcp = FastMCP(name="Restaurant Booking")


class BookingDetails(BaseModel):
    """Schema for restaurant booking information."""

    restaurant: str = Field(description="Choose a restaurant")
    party_size: int = Field(description="Number of people", ge=1, le=8)
    date: str = Field(description="Reservation date (DD-MM-YYYY)")


@mcp.tool()
async def book_table(ctx: Context) -> str:
    """Book a restaurant table with retry on validation errors."""
    while True:
        result = await ctx.elicit(
            message="Please provide your booking details:",
            schema=BookingDetails,
        )

        if result.action != "accept" or not result.data:
            return "Booking cancelled."

        try:
            # ✅ Handle both dict and BookingDetails case
            if isinstance(result.data, BookingDetails):
                booking = result.data
            else:
                booking = BookingDetails(**result.data)

            return f"✅ Booked table for {booking.party_size} at {booking.restaurant} on {booking.date}"
        except ValidationError as e:
            await ctx.notify("❌ Invalid booking details: " + str(e.errors()[0]["msg"]))
            continue



if __name__ == "__main__":
    print("🚀 Restaurant MCP server started, waiting for requests...")
    mcp.run(transport="stdio")

from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from load_models import MODEL


class Deps(BaseModel):
    """Default Dependencies"""
    weather_api_key: str | None = Field(title='Weather API Key', description='weather service API key')
    geo_api_key: str | None = Field(title='Geo API Key', description='geo service API key')
    

weather_agent = Agent(
    name='Weather Agent',
    model=MODEL,
    system_prompt=(
        'Be consice, reply with one sentence.'
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations,'
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    # result_type=JSON
)


@weather_agent.tool
def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> dict[str, float]:
    """
    Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of the location.
    """
    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key 
    }
    print('params passed by the agent:', params)
    # simulate the API call
    if 'London' in location_description:
        return {'lat': 10.795320, 'lng': -55.393958}
    else:
        raise ModelRetry('Could not find the location')
    

@weather_agent.tool
def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """
    Get the weather at a location.

    Args:
        ctx: The context.
        lat: The latitude of a location.
        lng: The longitude of a location.
    """
    if lat == 0 and lng == 0:
        raise ModelRetry('Coulg not find the location')
    
    params = {
        'lat': lat,
        'lng': lng,
        'api_key': ctx.deps.weather_api_key
    }
    # simulate an API call to get the latitude and longitude
    if lat == 10.795320 and lng == -55.393958:
        return {'temp':70, 'description':'windy'}
    

        
if __name__ == '__main__':
    deps = Deps(weather_api_key='<weather_api_key>', geo_api_key='<geo_api_key>')

    result = weather_agent.run_sync(
        'What is the weather like in London?',
        deps=deps
    )


    print('-----')
    print('Result:')
    print(result.data)


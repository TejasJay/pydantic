from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from dotenv import load_dotenv
import os

load_dotenv(override=True)

google_provider = GoogleProvider(api_key=os.getenv('GOOGLE_API_KEY'))
MODEL = GoogleModel('gemini-2.5-pro', provider=google_provider)


from app.core.config import get_settings
settings = get_settings()
print(settings.openai_api_key)  # reads from your .env file

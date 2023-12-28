from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    youtube_api: str
    openai_apikey: str
    google_custom_search_apikey: str
    search_engine_id: str

    class Config:
        env_file = ".env"

settings = Settings()
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    app_name: str = "RAG Application"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    
    # Database settings
    db_server: str = os.getenv("DB_SERVER", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "1433"))
    db_user: str = os.getenv("DB_USER", "sa")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "RagDatabase")
    
    # OpenAI API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Azure OpenAI 設置
    use_azure: bool = os.getenv("USE_AZURE", "True").lower() in ("true", "1", "t")
    deployment_name: str = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    azure_endpoint: str = os.getenv("AZURE_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    
    # RAG settings
    model_name: str = os.getenv("MODEL_NAME", "gpt-4-turbo")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    # LangChain chunking defaults
    default_chunk_size: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    default_chunk_overlap: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))

    # Connection string for SQL Server
    @property
    def db_connection_string(self) -> str:
        return f"mssql+pyodbc://{self.db_user}:{self.db_password}@{self.db_server}:{self.db_port}/{self.db_name}?driver=ODBC+Driver+18+for+SQL+Server"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Settings: Application settings.
    """
    return Settings()
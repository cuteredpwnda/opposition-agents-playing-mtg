"""Environment config — loaded from .env or environment variables."""

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        class BaseSettings:
            """Minimal fallback BaseSettings for environments without pydantic."""

            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        # With fallback BaseSettings, Settings class defaults work as plain dataclass.


class Settings(BaseSettings):
    """Application settings resolved from environment / .env file."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mtgpassword"

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm_provider: str = "openai"  # openai | anthropic | ollama
    default_llm_model: str = "gpt-4o"

    # Scryfall
    scryfall_cache_db: str = "data/card_cache.db"

    # Paths
    ontology_path: str = "data/ontology/mtg-ontology-v1.0.owl"
    shacl_shapes_path: str = "data/ontology/mtg-shapes.ttl"
    comprehensive_rules_path: str = "data/comprehensive_rules.txt"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

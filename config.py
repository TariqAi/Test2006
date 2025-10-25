import os
from typing import List
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration for deployment environments."""

    def __init__(self):
        # Detect environment
        self.environment = os.getenv("ENVIRONMENT", "production" if os.getenv("RENDER") else "development")
        self.is_production = os.getenv("RENDER") is not None or self.environment == "production"

        # Server configuration
        self.port = int(os.getenv("PORT", 10000))  # Render automatically provides PORT
        self.host = "0.0.0.0"

        # Production URL (Render deployment)
        self.production_url = "https://space-assistant-rag-system.onrender.com"
        self.render_external_url = os.getenv("RENDER_EXTERNAL_URL", self.production_url)

        # Local development URL
        self.local_url = f"http://localhost:{self.port}"

        # API base paths
        self.api_prefix = "/api"
        self.docs_url = "/api/docs"
        self.redoc_url = "/api/redoc"

        # CORS setup
        self.allowed_origins = self._get_allowed_origins()

        # Feature flags
        self.enable_docs = True
        self.enable_reload = not self.is_production

        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Database paths
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./enhanced_chroma_db")
        self.knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "space_science_knowledge_base.json")

        # Rate limiting
        self.tts_cooldown = int(os.getenv("TTS_COOLDOWN", "2"))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "info" if self.is_production else "debug")

    def _get_allowed_origins(self) -> List[str]:
        """CORS origins setup"""
        if self.is_production:
            origins = [
                self.production_url,
                self.render_external_url,
                "https://space-assistant-rag-system.onrender.com",
            ]
            custom_domain = os.getenv("CUSTOM_DOMAIN")
            if custom_domain:
                origins.append(f"https://{custom_domain}")
            return list(set(filter(None, origins)))
        else:
            return [
                "http://localhost:10000",
                "http://127.0.0.1:10000",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "*",
            ]

    def get_api_url(self) -> str:
        return self.render_external_url if self.is_production else self.local_url

    def get_full_url(self, path: str = "") -> str:
        return f"{self.get_api_url()}{path}"

    def print_config_info(self):
        print("\n" + "=" * 60)
        print("🚀 DEPLOYMENT CONFIGURATION")
        print("=" * 60)
        print(f"Environment:        {self.environment}")
        print(f"Production Mode:    {self.is_production}")
        print(f"API URL:            {self.get_api_url()}")
        print(f"Port:               {self.port}")
        print(f"Host:               {self.host}")
        print(f"Docs URL:           {self.get_full_url(self.docs_url)}")
        print(f"ReDoc URL:          {self.get_full_url(self.redoc_url)}")
        print(f"Reload Enabled:     {self.enable_reload}")
        print(f"Log Level:          {self.log_level}")
        print("\n🌍 Allowed Origins:")
        for o in self.allowed_origins:
            print(f"  • {o}")
        print("\n🔑 API KEYS:")
        print(f"OpenAI Key:         {'✅ Set' if self.openai_api_key else '❌ Missing'}")
        print(f"ElevenLabs Key:     {'✅ Set' if self.elevenlabs_api_key else '❌ Missing'}")
        print("=" * 60 + "\n")


# Create global instance
deployment_config = DeploymentConfig()


def setup_deployment_config() -> DeploymentConfig:
    config = DeploymentConfig()
    config.print_config_info()
    return config


def get_api_base_url() -> str:
    return deployment_config.get_api_url()


def get_cors_origins() -> List[str]:
    return deployment_config.allowed_origins


# Test run
if __name__ == "__main__":
    print("🔧 Testing Render Deployment Config...\n")
    config = setup_deployment_config()
    print(f"Base URL: {config.get_api_url()}")
    print(f"Docs:     {config.get_full_url('/api/docs')}")
    print(f"Health:   {config.get_full_url('/health')}")

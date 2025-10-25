#!/usr/bin/env python3
"""
Deployment Configuration for Space Science Assistant
Manages environment-specific settings for local development and production deployment.
"""

import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration for deployment environments."""
    
    def __init__(self):
        """Initialize deployment configuration."""
        # Environment detection
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = os.getenv("RENDER", False) or self.environment == "production"
        
        # Server configuration
        self.port = int(os.getenv("PORT", 10000))
        self.host = "0.0.0.0"
        
        # Production URL (Render.com)
        self.production_url = "https://space-assistant-rag-system.onrender.com"
        
        # Get Render external URL if available (Render sets this automatically)
        self.render_external_url = os.getenv("RENDER_EXTERNAL_URL", self.production_url)
        
        # Local development URL
        self.local_url = f"http://localhost:{self.port}"
        
        # API settings
        self.api_prefix = "/api"
        self.docs_url = "/api/docs"
        self.redoc_url = "/api/redoc"
        
        # CORS settings
        self.allowed_origins = self._get_allowed_origins()
        
        # Feature flags
        self.enable_docs = True  # Set to False in production if needed
        self.enable_reload = not self.is_production
        
        # API Keys (loaded from environment)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Database configuration
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./enhanced_chroma_db")
        self.knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "space_science_knowledge_base.json")
        
        # Rate limiting
        self.tts_cooldown = int(os.getenv("TTS_COOLDOWN", "2"))
        self.max_requests_per_minute = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "info" if self.is_production else "debug")
    
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed CORS origins based on environment."""
        if self.is_production:
            # Production: Allow only specific origins
            origins = [
                self.production_url,
                self.render_external_url,
                "https://space-assistant-rag-system.onrender.com"
            ]
            
            # Add custom domain if provided
            custom_domain = os.getenv("CUSTOM_DOMAIN")
            if custom_domain:
                origins.append(f"https://{custom_domain}")
            
            # Remove duplicates and None values
            origins = list(set(filter(None, origins)))
            
            return origins
        else:
            # Development: Allow all origins
            return [
                "http://localhost:10000",
                "http://localhost:3000",  # For frontend development
                "http://127.0.0.1:10000",
                "http://127.0.0.1:3000",
                "*"  # Allow all for development
            ]
    
    def get_api_url(self) -> str:
        """Get the appropriate API URL based on environment."""
        if self.is_production:
            return self.render_external_url or self.production_url
        return self.local_url
    
    def get_full_url(self, path: str = "") -> str:
        """Get full URL with path."""
        base_url = self.get_api_url()
        return f"{base_url}{path}"
    
    def get_allowed_origins(self) -> List[str]:
        """Get allowed CORS origins."""
        return self.allowed_origins
    
    def print_config_info(self):
        """Print configuration information for debugging."""
        print("\n" + "=" * 60)
        print("🔧 DEPLOYMENT CONFIGURATION")
        print("=" * 60)
        print(f"Environment:        {self.environment}")
        print(f"Is Production:      {self.is_production}")
        print(f"API URL:            {self.get_api_url()}")
        print(f"Port:               {self.port}")
        print(f"Host:               {self.host}")
        print(f"Docs URL:           {self.get_full_url(self.docs_url)}")
        print(f"ReDoc URL:          {self.get_full_url(self.redoc_url)}")
        print(f"Auto-reload:        {self.enable_reload}")
        print(f"Log Level:          {self.log_level}")
        print("\n🔐 API KEYS")
        print(f"OpenAI:             {'✅ Set' if self.openai_api_key else '❌ Missing'}")
        print(f"ElevenLabs:         {'✅ Set' if self.elevenlabs_api_key else '❌ Missing'}")
        print("\n🌐 CORS ORIGINS")
        for origin in self.allowed_origins:
            print(f"  • {origin}")
        print("\n📁 PATHS")
        print(f"Knowledge Base:     {self.knowledge_base_path}")
        print(f"ChromaDB:           {self.chroma_persist_dir}")
        print("=" * 60 + "\n")
    
    def validate_config(self) -> dict:
        """Validate configuration and return status."""
        issues = []
        warnings = []
        
        # Check API keys
        if not self.openai_api_key:
            issues.append("OpenAI API key not set (OPENAI_API_KEY)")
        
        if not self.elevenlabs_api_key:
            warnings.append("ElevenLabs API key not set (voice features disabled)")
        
        # Check file paths
        if not os.path.exists(self.knowledge_base_path):
            issues.append(f"Knowledge base not found: {self.knowledge_base_path}")
        
        # Check production settings
        if self.is_production:
            if "*" in self.allowed_origins:
                warnings.append("CORS allows all origins in production (security risk)")
            
            if self.enable_reload:
                warnings.append("Auto-reload enabled in production")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "environment": self.environment,
            "production": self.is_production
        }


# Create global configuration instance
deployment_config = DeploymentConfig()


def setup_deployment_config() -> DeploymentConfig:
    """Setup and validate deployment configuration."""
    config = DeploymentConfig()
    
    # Print configuration info
    config.print_config_info()
    
    # Validate configuration
    validation = config.validate_config()
    
    if not validation["valid"]:
        print("❌ CONFIGURATION ERRORS:")
        for issue in validation["issues"]:
            print(f"  • {issue}")
        print()
    
    if validation["warnings"]:
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")
        print()
    
    if validation["valid"]:
        print("✅ Configuration validated successfully!\n")
    else:
        print("⚠️  Configuration has errors. Please fix before deploying.\n")
    
    return config


# Environment-specific helper functions
def is_local() -> bool:
    """Check if running locally."""
    return not deployment_config.is_production


def is_production() -> bool:
    """Check if running in production."""
    return deployment_config.is_production


def get_api_base_url() -> str:
    """Get the base API URL."""
    return deployment_config.get_api_url()


def get_cors_origins() -> List[str]:
    """Get CORS allowed origins."""
    return deployment_config.get_allowed_origins()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Deployment Configuration...")
    config = setup_deployment_config()
    
    # Test URL generation
    print("\n🔗 URL EXAMPLES:")
    print(f"Base URL:           {config.get_api_url()}")
    print(f"API Docs:           {config.get_full_url('/api/docs')}")
    print(f"Health Check:       {config.get_full_url('/health')}")
    print(f"Ask Endpoint:       {config.get_full_url('/ask')}")
    
    # Test environment detection
    print("\n🔍 ENVIRONMENT DETECTION:")
    print(f"Environment:        {config.environment}")
    print(f"Is Production:      {config.is_production}")
    print(f"Is Local:           {is_local()}")
    
    # Run validation
    validation = config.validate_config()
    print("\n✅ VALIDATION RESULT:")
    print(f"Valid:              {validation['valid']}")
    if validation['issues']:
        print(f"Issues:             {len(validation['issues'])}")
    if validation['warnings']:
        print(f"Warnings:           {len(validation['warnings'])}")

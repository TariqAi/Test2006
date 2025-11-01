#!/usr/bin/env python3
"""
Configuration Management for Space Science Assistant
Manages API keys, settings, and environment variables for all integrated services.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class APIConfig:
    """Configuration for API services."""
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    
    # Model settings
    gpt_model: str = "gpt-4"
    whisper_model: str = "whisper-1"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Voice settings
    elevenlabs_voice_id: Optional[str] = None
    elevenlabs_model: str = "eleven_monolingual_v1"
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.elevenlabs_api_key:
            self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    # Recording settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    max_record_seconds: int = 10
    
    # Audio format
    audio_format: str = "paInt16"  # PyAudio format
    
    # Voice activity detection
    silence_threshold: float = 0.01
    silence_duration: float = 2.0  # seconds of silence to stop recording

@dataclass
class AssistantConfig:
    """Configuration for the assistant behavior."""
    # Knowledge base
    knowledge_base_path: str = "space_science_knowledge_base.json"
    
    # ChromaDB settings
    chroma_persist_dir: str = "./enhanced_chroma_db"
    collection_name: str = "enhanced_space_science_kb"
    
    # Response settings
    max_response_length: int = 800
    max_tts_length: int = 500  # Shorter for better TTS
    max_retrieved_chunks: int = 5
    
    # Conversation settings
    max_conversation_history: int = 50
    context_window_size: int = 10  # Number of previous exchanges to consider

class ConfigManager:
    """Manages configuration for the Space Science Assistant."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file or ".env"
        
        # Load configurations
        self.api = APIConfig()
        self.audio = AudioConfig()
        self.assistant = AssistantConfig()
        
        # Load from file if exists
        self._load_from_file()
    
    def _load_from_file(self):
        """Load configuration from .env file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"\'')
                
                # Reload API config with new environment variables
                self.api = APIConfig()
                
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def create_env_template(self, filename: str = ".env.template"):
        """Create a template .env file with all required settings."""
        template_content = """
# Space Science Assistant Configuration
# Copy this file to .env and fill in your API keys

# =============================================================================
# API KEYS (Required)
# =============================================================================

# OpenAI API Key (for GPT-4 and Whisper)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# ElevenLabs API Key (for Text-to-Speech)
# Get from: https://elevenlabs.io/
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# =============================================================================
# OPTIONAL SETTINGS
# =============================================================================

# ElevenLabs Voice ID (optional - will use default if not set)
# Find voice IDs at: https://elevenlabs.io/voices
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Model Settings
# GPT_MODEL=gpt-4
# WHISPER_MODEL=whisper-1
# EMBEDDING_MODEL=all-MiniLM-L6-v2

# Audio Settings
# SAMPLE_RATE=16000
# MAX_RECORD_SECONDS=10

# Assistant Settings
# MAX_RESPONSE_LENGTH=800
# MAX_RETRIEVED_CHUNKS=5
"""
        
        with open(filename, 'w') as f:
            f.write(template_content.strip())
        
        print(f"✅ Created configuration template: {filename}")
        print(f"📝 Please copy to .env and add your API keys")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        status = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'features_available': {
                'gpt4': False,
                'tts': False,
                'stt': False,
                'voice_interface': False
            }
        }
        
        # Check OpenAI API key
        if not self.api.openai_api_key:
            status['errors'].append("OpenAI API key not found")
            status['valid'] = False
        else:
            status['features_available']['gpt4'] = True
            status['features_available']['stt'] = True
        
        # Check ElevenLabs API key
        if not self.api.elevenlabs_api_key:
            status['warnings'].append("ElevenLabs API key not found - TTS disabled")
        else:
            status['features_available']['tts'] = True
        
        # Voice interface requires both
        if status['features_available']['tts'] and status['features_available']['stt']:
            status['features_available']['voice_interface'] = True
        
        # Check file paths
        if not os.path.exists(self.assistant.knowledge_base_path):
            status['errors'].append(f"Knowledge base not found: {self.assistant.knowledge_base_path}")
            status['valid'] = False
        
        return status
    
    def print_status(self):
        """Print configuration status."""
        status = self.validate_config()
        
        print("\n" + "="*50)
        print("🔧 SPACE SCIENCE ASSISTANT CONFIGURATION")
        print("="*50)
        
        # API Keys Status
        print("\n🔑 API Keys:")
        print(f"  OpenAI (GPT-4/Whisper): {'✅ Set' if self.api.openai_api_key else '❌ Missing'}")
        print(f"  ElevenLabs (TTS):       {'✅ Set' if self.api.elevenlabs_api_key else '❌ Missing'}")
        
        # Features Status
        print("\n🚀 Available Features:")
        features = status['features_available']
        print(f"  GPT-4 Responses:        {'✅ Available' if features['gpt4'] else '❌ Disabled'}")
        print(f"  Text-to-Speech:         {'✅ Available' if features['tts'] else '❌ Disabled'}")
        print(f"  Speech-to-Text:         {'✅ Available' if features['stt'] else '❌ Disabled'}")
        print(f"  Voice Interface:        {'✅ Available' if features['voice_interface'] else '❌ Disabled'}")
        
        # Model Settings
        print("\n⚙️  Model Settings:")
        print(f"  GPT Model:              {self.api.gpt_model}")
        print(f"  Whisper Model:          {self.api.whisper_model}")
        print(f"  Embedding Model:        {self.api.embedding_model}")
        
        # Warnings and Errors
        if status['warnings']:
            print("\n⚠️  Warnings:")
            for warning in status['warnings']:
                print(f"  • {warning}")
        
        if status['errors']:
            print("\n❌ Errors:")
            for error in status['errors']:
                print(f"  • {error}")
        
        # Overall Status
        print("\n" + "="*50)
        if status['valid']:
            print("✅ Configuration is valid!")
        else:
            print("❌ Configuration has errors - please fix before using")
        
        print("="*50)
        
        return status
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for missing components."""
        status = self.validate_config()
        instructions = []
        
        if not self.api.openai_api_key:
            instructions.append("""
🔑 OpenAI API Key Setup:
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to .env file: OPENAI_API_KEY=your_key_here
""")
        
        if not self.api.elevenlabs_api_key:
            instructions.append("""
🔊 ElevenLabs API Key Setup:
1. Go to https://elevenlabs.io/
2. Sign up for an account
3. Get your API key from settings
4. Add to .env file: ELEVENLABS_API_KEY=your_key_here
""")
        
        if not os.path.exists(self.assistant.knowledge_base_path):
            instructions.append("""
📚 Knowledge Base Setup:
1. Ensure space_science_knowledge_base.json exists
2. Run the knowledge base creation script if needed
""")
        
        return "\n".join(instructions) if instructions else "✅ All components are properly configured!"

# Global configuration instance
config = ConfigManager()

def setup_assistant_config() -> ConfigManager:
    """Setup and validate assistant configuration."""
    print("🔧 Setting up Space Science Assistant configuration...")
    
    # Create .env template if it doesn't exist
    if not os.path.exists(".env") and not os.path.exists(".env.template"):
        config.create_env_template()
    
    # Print status
    status = config.print_status()
    
    # Show setup instructions if needed
    if not status['valid'] or status['warnings']:
        print("\n📋 Setup Instructions:")
        print(config.get_setup_instructions())
    
    return config

if __name__ == "__main__":
    # Run configuration setup
    setup_assistant_config()
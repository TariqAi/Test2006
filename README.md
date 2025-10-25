# 🌌 Space Science AI Assistant

An advanced Retrieval-Augmented Generation (RAG) system for space science education, powered by GPT-4, ChromaDB, ElevenLabs TTS, and OpenAI Whisper STT.

## 🚀 Features

### Core Capabilities
- **📚 Knowledge Base**: Comprehensive space science information with semantic search
- **🤖 GPT-4 Integration**: Advanced natural language understanding and response generation
- **🔍 Vector Search**: ChromaDB-powered similarity search with embeddings
- **💬 Conversational AI**: Context-aware dialogue with conversation history
- **🎤 Voice Interface**: Speech-to-text input and text-to-speech output
- **📖 Source Citations**: Transparent knowledge sourcing and fact verification

### Advanced Features
- **🧠 Query Optimization**: Intelligent query enhancement for better retrieval
- **🏷️ Topic Classification**: Automatic categorization of questions and responses
- **📊 Session Management**: Conversation history and interaction tracking
- **⚙️ Flexible Configuration**: Easy API key and settings management
- **🔧 Modular Architecture**: Extensible component-based design

## 📋 Requirements

### System Requirements
- Python 3.8+
- Windows/macOS/Linux
- Internet connection for API calls

### API Keys (Required)
- **OpenAI API Key**: For GPT-4 and Whisper STT
- **ElevenLabs API Key**: For Text-to-Speech (optional)

## 🛠️ Installation

### 1. Clone or Download
```bash
# If using git
git clone <repository-url>
cd space-science-assistant

# Or download and extract the files to a directory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
# Run configuration setup
python config.py

# This creates a .env.template file
# Copy it to .env and add your API keys:
cp .env.template .env
```

Edit the `.env` file:
```env
# Required for GPT-4 and Whisper
OPENAI_API_KEY=your_openai_api_key_here

# Optional for Text-to-Speech
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

### 4. Verify Setup
```bash
python config.py
```

## 🎯 Usage

### Quick Start
```bash
# Run the main assistant
python main_assistant.py
```

### Available Interfaces

#### 1. Main Assistant (Recommended)
```bash
python main_assistant.py
```
- Full-featured interface with all capabilities
- Voice mode support
- Session management
- Configuration validation

#### 2. Interactive Text Assistant
```bash
python interactive_assistant.py
```
- Text-only interface
- Basic RAG functionality
- Good for testing

#### 3. Voice Assistant
```bash
python voice_assistant.py
```
- Voice-only interface
- Requires both OpenAI and ElevenLabs API keys

#### 4. Enhanced Assistant (Programmatic)
```python
from enhanced_space_assistant import EnhancedSpaceScienceAssistant

assistant = EnhancedSpaceScienceAssistant()
assistant.initialize()
response = assistant.ask_question("What is a black hole?")
print(response['answer'])
```

## 💬 Example Interactions

### Text Mode
```
🌌 You: What is a black hole?

🤖 Assistant: A black hole is a region in spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the event horizon. Black holes form when massive stars collapse at the end of their lives...

📚 Sources: Stellar Astronomy, Space Phenomena
🏷️ Topics: Black Holes, Stellar Evolution
⏱️ Response time: 1.23s
```

### Voice Mode
```
🎤 Voice Mode Activated!
   Speak your questions naturally
   Say 'stop voice mode' to return to text

🎙️ Listening... (speak now)
🔍 Processing: "Tell me about the James Webb Space Telescope"
🤖 Generating response...
🔊 Speaking response...
```

## 📁 Project Structure

```
space-science-assistant/
├── 📄 main_assistant.py          # Main application entry point
├── 🤖 enhanced_space_assistant.py # GPT-4 enhanced RAG system
├── 🎤 voice_assistant.py          # Voice interface components
├── 💬 interactive_assistant.py    # Text-only interface
├── ⚙️ config.py                   # Configuration management
├── 🔍 query_optimizer.py          # Query enhancement system
├── 📚 space_science_knowledge_base.json # Knowledge database
├── 📋 requirements.txt            # Python dependencies
├── 🔧 .env.template              # Configuration template
├── 📖 README.md                  # This file
└── 📁 Generated Files/
    ├── chroma_db/                # ChromaDB vector database
    ├── enhanced_chroma_db/       # Enhanced database
    └── __pycache__/             # Python cache
```

## 🔧 Configuration Options

### Environment Variables
```env
# API Keys
OPENAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here

# Model Settings
GPT_MODEL=gpt-4
WHISPER_MODEL=whisper-1
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Voice Settings
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
SAMPLE_RATE=16000
MAX_RECORD_SECONDS=10

# Response Settings
MAX_RESPONSE_LENGTH=800
MAX_RETRIEVED_CHUNKS=5
```

### Available Commands
- `help` - Show detailed help
- `voice` - Start voice interaction
- `topics` - List available topics
- `history` - Show conversation history
- `clear` - Clear conversation history
- `status` - Show system status
- `quit` - Exit the assistant

## 📖 Available Topics

The knowledge base covers:
1. **Latest Discoveries** - Recent space science findings
2. **Mars Exploration** - Rovers, missions, and discoveries
3. **Planetary Science** - Solar system bodies and characteristics
4. **Space Missions** - Current and historical space missions
5. **Space Phenomena** - Black holes, neutron stars, etc.
6. **Stellar Astronomy** - Stars, galaxies, and cosmic structures

## 🔍 How It Works

### RAG Pipeline
1. **Query Processing**: User input is optimized and enhanced
2. **Vector Search**: ChromaDB finds relevant knowledge chunks
3. **Context Assembly**: Retrieved information is organized
4. **Response Generation**: GPT-4 creates comprehensive answers
5. **Source Citation**: References are provided for transparency

### Voice Pipeline
1. **Speech Capture**: Audio recording with silence detection
2. **Speech-to-Text**: OpenAI Whisper converts audio to text
3. **RAG Processing**: Standard RAG pipeline processes the query
4. **Text-to-Speech**: ElevenLabs converts response to audio
5. **Audio Playback**: Response is played through speakers

## 🚨 Troubleshooting

### Common Issues

#### "Configuration has errors"
```bash
# Check your configuration
python config.py

# Ensure API keys are set in .env file
```

#### "Voice mode not available"
- Verify both OpenAI and ElevenLabs API keys are configured
- Check audio device permissions
- Install audio dependencies: `pip install pyaudio`

#### "ChromaDB initialization failed"
- Delete the `chroma_db/` directory and restart
- Check file permissions
- Ensure sufficient disk space

#### "Import errors"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Getting Help
1. Run `python config.py` to check configuration
2. Use the `help` command within the assistant
3. Check the console output for detailed error messages

## 🔐 Security Notes

- **API Keys**: Never commit `.env` files to version control
- **Local Storage**: All data is stored locally (ChromaDB, conversation history)
- **Network**: Only API calls to OpenAI and ElevenLabs are made
- **Privacy**: No conversation data is sent to third parties beyond API providers

## 🎨 Customization

### Adding Knowledge
Edit `space_science_knowledge_base.json` to add new topics:
```json
{
  "title": "New Topic",
  "content": "Detailed information...",
  "source": "Source Name",
  "topics": ["Topic1", "Topic2"]
}
```

### Modifying Responses
Adjust settings in `config.py`:
- `max_response_length`: Control response verbosity
- `max_retrieved_chunks`: Number of knowledge pieces to use
- `max_tts_length`: Limit spoken response length

### Voice Customization
- Change ElevenLabs voice ID in `.env`
- Adjust audio settings (sample rate, recording duration)
- Modify silence detection thresholds

## 📊 Performance

### Typical Response Times
- **Text Query**: 1-3 seconds
- **Voice Query**: 3-8 seconds (including STT/TTS)
- **Knowledge Base Search**: <1 second
- **GPT-4 Generation**: 1-5 seconds

### Resource Usage
- **Memory**: ~500MB (with embeddings loaded)
- **Storage**: ~50MB (ChromaDB + models)
- **Network**: API calls only

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Clearing Cache
```bash
# Remove ChromaDB (will rebuild on next run)
rm -rf chroma_db/ enhanced_chroma_db/

# Clear Python cache
rm -rf __pycache__/
```

### Backup
Important files to backup:
- `.env` (API keys)
- `space_science_knowledge_base.json` (if modified)
- Conversation history (stored in ChromaDB)

## 🤝 Contributing

To extend the assistant:
1. Add new knowledge to the JSON file
2. Implement new features in modular components
3. Update configuration options as needed
4. Test with various query types

## 📄 License

This project is for educational purposes. Please respect API terms of service for OpenAI and ElevenLabs.

---

**🌟 Ready to explore the cosmos? Run `python main_assistant.py` to get started!**
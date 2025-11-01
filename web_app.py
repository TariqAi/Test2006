from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import io
import wave
import time
from threading import Lock
from enhanced_space_assistant import EnhancedSpaceScienceAssistant
from contextlib import asynccontextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("⚠️  Voice libraries not available. Install with: pip install openai elevenlabs")

# Define Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    question: str

# Setup lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    global space_assistant
    print("🚀 Initializing Space Science Assistant...")
    try:
        space_assistant = EnhancedSpaceScienceAssistant()
        print("✅ Space Science Assistant initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Error initializing assistant: {e}")
        print("   The service will continue, but some features may not work.")
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down Space Science Assistant...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Space Science AI Assistant",
    description="Advanced AI-powered space science knowledge system with voice capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; in production, specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Rate limiting for TTS requests
tts_lock = Lock()
tts_last_request = {}
TTS_COOLDOWN = 2  # seconds between requests per session

# Mount static files if they exist
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    """
    Ask the Space Science Assistant a question.
    
    - **question**: The question to ask about space science
    """
    try:
        question = question_request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question")
        
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        return {
            "response": response.get('response', ''),
            "sources": response.get('sources', []),
            "topics": response.get('topics', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/topics")
async def get_topics():
    """
    Get all available topics in the knowledge base.
    """
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        topics = space_assistant.get_available_topics()
        return {"topics": topics}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Define models for file uploads and responses
class SpeechToTextResponse(BaseModel):
    text: str
    success: bool

class TextToSpeechRequest(BaseModel):
    text: str

class TextToSpeechResponse(BaseModel):
    audio: str
    success: bool

@app.post("/speech_to_text", response_model=SpeechToTextResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert uploaded audio to text using OpenAI Whisper.
    
    - **audio**: Audio file (WAV format recommended)
    """
    try:
        if not VOICE_ENABLED:
            raise HTTPException(status_code=400, detail="Voice features not available")
        
        if not audio:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Use OpenAI Whisper for speech-to-text
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")
        
        client = openai.OpenAI(api_key=openai_key)
        
        # Read audio file content
        audio_content = await audio.read()
        
        # Convert to OpenAI format
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.wav", audio_content, "audio/wav"),
            language="en"
        )
        
        return SpeechToTextResponse(
            text=transcript.text.strip(),
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {str(e)}")

@app.post("/text_to_speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest, req: Request):
    """
    Convert text to speech using ElevenLabs.
    
    - **text**: The text to convert to speech
    """
    try:
        if not VOICE_ENABLED:
            raise HTTPException(status_code=400, detail="Voice features not available")
        
        # Simple rate limiting based on IP
        client_ip = req.client.host
        current_time = time.time()
        
        with tts_lock:
            if client_ip in tts_last_request:
                time_since_last = current_time - tts_last_request[client_ip]
                if time_since_last < TTS_COOLDOWN:
                    raise HTTPException(
                        status_code=429,
                        detail={
                            'error': f'Please wait {TTS_COOLDOWN - time_since_last:.1f} seconds before making another request.',
                            'rate_limited': True
                        }
                    )
            
            tts_last_request[client_ip] = current_time
        
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Setup ElevenLabs API key
        # Setup ElevenLabs API key
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=elevenlabs_key)
        
        # Generate audio using ElevenLabs
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Convert audio to base64 for web transmission
        import base64
        # ElevenLabs returns an iterator of audio chunks
        audio_bytes = b''.join(audio)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return TextToSpeechResponse(
            audio=audio_base64,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Handle ElevenLabs rate limiting
        if '429' in error_msg or 'too_many_concurrent_requests' in error_msg:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'ElevenLabs rate limit exceeded. Please wait a moment and try again.',
                    'rate_limited': True
                }
            )
        raise HTTPException(status_code=500, detail=f"Text-to-speech error: {error_msg}")

class VoiceStatusResponse(BaseModel):
    voice_enabled: bool
    openai_key: bool
    elevenlabs_key: bool

@app.get("/voice_status", response_model=VoiceStatusResponse)
async def voice_status():
    """
    Check if voice features are available.
    """
    return VoiceStatusResponse(
        voice_enabled=VOICE_ENABLED,
        openai_key=bool(os.getenv('OPENAI_API_KEY')),
        elevenlabs_key=bool(os.getenv('ELEVENLABS_API_KEY'))
    )

class HistoryResponse(BaseModel):
    history: list

@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """
    Get conversation history.
    """
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        history = space_assistant.get_conversation_history()
        return HistoryResponse(history=history)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class MessageResponse(BaseModel):
    message: str

@app.post("/clear_history", response_model=MessageResponse)
async def clear_history():
    """
    Clear conversation history.
    """
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        space_assistant.clear_conversation_history()
        return MessageResponse(message="Conversation history cleared")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class RebuildResponse(BaseModel):
    success: bool
    message: str

@app.post("/rebuild_knowledge", response_model=RebuildResponse)
async def rebuild_knowledge():
    """
    Rebuild the knowledge base with updated information.
    """
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        space_assistant.rebuild_knowledge_base()
        return RebuildResponse(success=True, message="Knowledge base rebuilt successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == '__main__':
    print("Starting Space Science Assistant Web Interface...")
    
    print("\n🚀 Space Science Assistant Web UI is ready!")
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get("PORT", 10000))
    
    print(f"🌟 Server will run on port: {port}")
    print("🌙 Explore the cosmos with our AI assistant!\n")
    
    # Run the FastAPI app with Uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )

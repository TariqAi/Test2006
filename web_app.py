#!/usr/bin/env python3
"""
FastAPI Web Application for Space Science Assistant
Converted from Flask with enhanced features and async support.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import time
import base64
from threading import Lock
from datetime import datetime
import uvicorn

from enhanced_space_assistant import EnhancedSpaceScienceAssistant

# Simple configuration - don't import complex config that might fail
class SimpleConfig:
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = os.getenv("RENDER", False) or self.environment == "production"
        self.port = int(os.getenv("PORT", 10000))
        self.render_url = "https://space-assistant-rag-system.onrender.com"
        
    def get_api_url(self):
        if self.is_production:
            return self.render_url
        return f"http://localhost:{self.port}"
    
    def get_allowed_origins(self):
        if self.is_production:
            return [self.render_url, "https://space-assistant-rag-system.onrender.com"]
        return ["*"]

config = SimpleConfig()

# Import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("⚠️  Voice libraries not available. Install with: pip install openai elevenlabs")

# Initialize FastAPI app
app = FastAPI(
    title="Space Science AI Assistant",
    description="Advanced AI-powered space science knowledge system with voice capabilities",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates (if you have HTML templates)
templates = Jinja2Templates(directory="templates")

# Mount static files (if you have them)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Rate limiting for TTS requests
tts_lock = Lock()
tts_last_request: Dict[str, float] = {}
TTS_COOLDOWN = 2  # seconds between requests per session

# Initialize the space assistant
space_assistant: Optional[EnhancedSpaceScienceAssistant] = None

# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    question: str

class TextToSpeechRequest(BaseModel):
    text: str

class QuestionResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    topics: List[str]

class TopicsResponse(BaseModel):
    topics: List[str]

class VoiceStatusResponse(BaseModel):
    voice_enabled: bool
    openai_key: bool
    elevenlabs_key: bool

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str
    rate_limited: Optional[bool] = None

class SpeechToTextResponse(BaseModel):
    text: str
    success: bool

class TextToSpeechResponse(BaseModel):
    audio: str
    success: bool

class RebuildResponse(BaseModel):
    success: bool
    message: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup."""
    global space_assistant
    print("🚀 Initializing Space Science Assistant...")
    try:
        space_assistant = EnhancedSpaceScienceAssistant()
        # Remove the extra initialize() call since __init__ already does it
        print("✅ Space Science Assistant initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Error initializing assistant: {e}")
        print("   The service will continue, but some features may not work.")
        # Don't exit - let the service run anyway

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🛑 Shutting down Space Science Assistant...")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    if os.path.exists("templates/index.html"):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "api_url": config.get_api_url()
        })
    else:
        api_url = config.get_api_url()
        return HTMLResponse(content=f"""
        <html>
            <head><title>Space Science Assistant</title></head>
            <body>
                <h1>🚀 Space Science AI Assistant</h1>
                <p>API is running at: <strong>{api_url}</strong></p>
                <p>Visit <a href="{api_url}/api/docs">/api/docs</a> for documentation.</p>
            </body>
        </html>
        """)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "assistant_initialized": space_assistant is not None,
        "voice_enabled": VOICE_ENABLED,
        "timestamp": datetime.now().isoformat(),
        "environment": config.environment
    }

# Main Q&A endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask the Space Science Assistant a question.
    
    - **question**: The question to ask about space science
    """
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question")
        
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        return QuestionResponse(
            response=response.get('response', ''),
            sources=response.get('sources', []),
            topics=response.get('topics_covered', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Topics endpoint
@app.get("/topics", response_model=TopicsResponse)
async def get_topics():
    """
    Get all available topics in the knowledge base.
    """
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        topics = space_assistant.get_available_topics()
        return TopicsResponse(topics=topics)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Speech to text endpoint
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

# Text to speech endpoint
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

# Voice status endpoint
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

# Conversation history endpoint
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

# Clear history endpoint
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

# Rebuild knowledge base endpoint
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Main entry point
if __name__ == '__main__':
    # ✅ استيراد إعدادات النشر من ملف deployment_config
    from deployment_config import setup_deployment_config
    
    # ✅ تحميل الإعدادات
    config = setup_deployment_config()

    # ✅ استخدم القيم من config بدل من القيم الثابتة
    port = config.port
    host = config.host

    print(f"🚀 Starting Space Science Assistant...")
    print(f"🌍 Environment: {config.environment}")
    print(f"🌍 Host: {host}")
    print(f"🌍 Port: {port}")
    print(f"🌍 API URL: {config.get_api_url()}")
    
    # ✅ تشغيل Uvicorn باستخدام الإعدادات الديناميكية
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=config.log_level,
        reload=config.enable_reload,
        access_log=True
    )

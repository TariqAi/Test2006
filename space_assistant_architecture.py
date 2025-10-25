"""Space Science AI Assistant Architecture

This module defines the core architecture and components for building
a comprehensive space science AI assistant.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Core Data Structures
@dataclass
class KnowledgeChunk:
    """Represents a single piece of knowledge in the database."""
    id: str
    text: str
    topic: str
    subtopic: str
    type: str  # Mission/Planet/Concept
    source: str
    year: str
    keywords: List[str]
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None

@dataclass
class UserQuery:
    """Represents a user's query with processing metadata."""
    original_text: str
    optimized_text: str
    keywords: List[str]
    topics: List[str]
    intent: str  # question, explanation, comparison, etc.
    timestamp: datetime

@dataclass
class ConversationContext:
    """Maintains conversation state and history."""
    session_id: str
    history: List[Dict[str, Any]]
    current_topic: Optional[str]
    user_preferences: Dict[str, Any]
    last_query: Optional[UserQuery]

@dataclass
class AssistantResponse:
    """Structured response from the AI assistant."""
    answer: str
    sources: List[str]
    confidence: float
    related_topics: List[str]
    follow_up_suggestions: List[str]
    chunks_used: List[KnowledgeChunk]

class QueryIntent(Enum):
    """Types of user query intents."""
    FACTUAL_QUESTION = "factual_question"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    LATEST_NEWS = "latest_news"
    MISSION_STATUS = "mission_status"
    DEFINITION = "definition"

# Abstract Base Classes
class QueryProcessor(ABC):
    """Abstract base class for query processing."""
    
    @abstractmethod
    def process_query(self, query: str) -> UserQuery:
        """Process raw user query into structured format."""
        pass
    
    @abstractmethod
    def detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of the user's query."""
        pass

class KnowledgeRetriever(ABC):
    """Abstract base class for knowledge retrieval."""
    
    @abstractmethod
    def search(self, query: UserQuery, top_k: int = 5) -> List[KnowledgeChunk]:
        """Search for relevant knowledge chunks."""
        pass
    
    @abstractmethod
    def get_by_topic(self, topic: str, limit: int = 10) -> List[KnowledgeChunk]:
        """Retrieve chunks by topic."""
        pass

class ResponseGenerator(ABC):
    """Abstract base class for response generation."""
    
    @abstractmethod
    def generate_response(self, 
                         query: UserQuery, 
                         chunks: List[KnowledgeChunk],
                         context: ConversationContext) -> AssistantResponse:
        """Generate a response based on query and retrieved knowledge."""
        pass

class ContextManager(ABC):
    """Abstract base class for conversation context management."""
    
    @abstractmethod
    def update_context(self, query: UserQuery, response: AssistantResponse) -> None:
        """Update conversation context with new interaction."""
        pass
    
    @abstractmethod
    def get_context(self, session_id: str) -> ConversationContext:
        """Retrieve conversation context for a session."""
        pass

# Main Assistant Architecture
class SpaceScienceAssistant:
    """Main AI Assistant class that orchestrates all components."""
    
    def __init__(self,
                 query_processor: QueryProcessor,
                 knowledge_retriever: KnowledgeRetriever,
                 response_generator: ResponseGenerator,
                 context_manager: ContextManager):
        self.query_processor = query_processor
        self.knowledge_retriever = knowledge_retriever
        self.response_generator = response_generator
        self.context_manager = context_manager
    
    def ask(self, question: str, session_id: str = "default") -> AssistantResponse:
        """Main method to ask the assistant a question."""
        
        # 1. Process the user query
        processed_query = self.query_processor.process_query(question)
        
        # 2. Get conversation context
        context = self.context_manager.get_context(session_id)
        
        # 3. Retrieve relevant knowledge
        relevant_chunks = self.knowledge_retriever.search(processed_query)
        
        # 4. Generate response
        response = self.response_generator.generate_response(
            processed_query, relevant_chunks, context
        )
        
        # 5. Update context
        self.context_manager.update_context(processed_query, response)
        
        return response
    
    def get_topic_overview(self, topic: str) -> Dict[str, Any]:
        """Get an overview of a specific topic."""
        chunks = self.knowledge_retriever.get_by_topic(topic)
        
        return {
            "topic": topic,
            "total_chunks": len(chunks),
            "subtopics": list(set(chunk.subtopic for chunk in chunks)),
            "sources": list(set(chunk.source for chunk in chunks)),
            "key_concepts": self._extract_key_concepts(chunks)
        }
    
    def _extract_key_concepts(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """Extract key concepts from a list of chunks."""
        all_keywords = []
        for chunk in chunks:
            all_keywords.extend(chunk.keywords)
        
        # Count frequency and return top concepts
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency and return top 10
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:10]]

# Configuration and Factory Classes
class AssistantConfig:
    """Configuration settings for the assistant."""
    
    def __init__(self):
        self.knowledge_base_path = "space_science_knowledge_base.json"
        self.vector_db_path = "./chroma_db"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.max_chunks_per_response = 5
        self.confidence_threshold = 0.7
        self.enable_citations = True
        self.enable_follow_ups = True

class AssistantFactory:
    """Factory class to create and configure the assistant."""
    
    @staticmethod
    def create_assistant(config: AssistantConfig) -> SpaceScienceAssistant:
        """Create a fully configured assistant instance."""
        
        # Import concrete implementations (to be created)
        from query_optimizer import SpaceScienceQueryOptimizer
        # from vector_search import ChromaKnowledgeRetriever
        # from response_gen import OpenAIResponseGenerator
        # from context_mgr import InMemoryContextManager
        
        # Create components
        query_processor = ConcreteQueryProcessor()
        knowledge_retriever = ConcreteKnowledgeRetriever(config)
        response_generator = ConcreteResponseGenerator(config)
        context_manager = ConcreteContextManager()
        
        return SpaceScienceAssistant(
            query_processor=query_processor,
            knowledge_retriever=knowledge_retriever,
            response_generator=response_generator,
            context_manager=context_manager
        )

# Placeholder concrete implementations (to be developed)
class ConcreteQueryProcessor(QueryProcessor):
    """Concrete implementation of query processing."""
    
    def __init__(self):
        from query_optimizer import SpaceScienceQueryOptimizer
        self.optimizer = SpaceScienceQueryOptimizer()
    
    def process_query(self, query: str) -> UserQuery:
        result = self.optimizer.optimize_query(query)
        return UserQuery(
            original_text=query,
            optimized_text=result.get('optimized_query', query),
            keywords=result.get('extracted_keywords', []),
            topics=result.get('relevant_topics', []),
            intent=self.detect_intent(query).value,
            timestamp=datetime.now()
        )
    
    def detect_intent(self, query: str) -> QueryIntent:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return QueryIntent.DEFINITION
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            return QueryIntent.EXPLANATION
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            return QueryIntent.COMPARISON
        elif any(word in query_lower for word in ['latest', 'recent', 'new', 'discovery']):
            return QueryIntent.LATEST_NEWS
        elif any(word in query_lower for word in ['mission', 'status', 'progress']):
            return QueryIntent.MISSION_STATUS
        else:
            return QueryIntent.FACTUAL_QUESTION

class ConcreteKnowledgeRetriever(KnowledgeRetriever):
    """Placeholder for vector database implementation."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        # TODO: Initialize Chroma DB connection
    
    def search(self, query: UserQuery, top_k: int = 5) -> List[KnowledgeChunk]:
        # TODO: Implement vector search
        return []
    
    def get_by_topic(self, topic: str, limit: int = 10) -> List[KnowledgeChunk]:
        # TODO: Implement topic-based retrieval
        return []

class ConcreteResponseGenerator(ResponseGenerator):
    """Placeholder for response generation implementation."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
    
    def generate_response(self, 
                         query: UserQuery, 
                         chunks: List[KnowledgeChunk],
                         context: ConversationContext) -> AssistantResponse:
        # TODO: Implement response generation
        return AssistantResponse(
            answer="Response generation not yet implemented.",
            sources=[],
            confidence=0.0,
            related_topics=[],
            follow_up_suggestions=[],
            chunks_used=[]
        )

class ConcreteContextManager(ContextManager):
    """Placeholder for context management implementation."""
    
    def __init__(self):
        self.contexts = {}
    
    def update_context(self, query: UserQuery, response: AssistantResponse) -> None:
        # TODO: Implement context updates
        pass
    
    def get_context(self, session_id: str) -> ConversationContext:
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                history=[],
                current_topic=None,
                user_preferences={},
                last_query=None
            )
        return self.contexts[session_id]

# Example usage and testing
if __name__ == "__main__":
    print("Space Science AI Assistant Architecture")
    print("=" * 40)
    
    # Create configuration
    config = AssistantConfig()
    
    # Create assistant (with placeholder implementations)
    assistant = AssistantFactory.create_assistant(config)
    
    print("\nArchitecture Components:")
    print(f"- Query Processor: {type(assistant.query_processor).__name__}")
    print(f"- Knowledge Retriever: {type(assistant.knowledge_retriever).__name__}")
    print(f"- Response Generator: {type(assistant.response_generator).__name__}")
    print(f"- Context Manager: {type(assistant.context_manager).__name__}")
    
    print("\nNext Steps:")
    print("1. Implement Chroma DB integration for vector search")
    print("2. Add embedding generation for semantic search")
    print("3. Create response generation with LLM integration")
    print("4. Build conversational interface (CLI/Web)")
    print("5. Add comprehensive testing and evaluation")
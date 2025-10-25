import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from query_optimizer import SpaceScienceQueryOptimizer
import os

@dataclass
class RetrievedChunk:
    """Represents a retrieved knowledge chunk with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    similarity_score: float

class SpaceScienceRAGAssistant:
    """RAG-based Space Science AI Assistant."""
    
    def __init__(self, 
                 knowledge_base_path: str = "space_science_knowledge_base.json",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chroma_persist_dir: str = "./chroma_db"):
        """
        Initialize the RAG assistant.
        
        Args:
            knowledge_base_path: Path to the JSON knowledge base
            embedding_model: Sentence transformer model for embeddings
            chroma_persist_dir: Directory to persist ChromaDB
        """
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model
        self.chroma_persist_dir = chroma_persist_dir
        
        # Initialize components
        self.query_optimizer = SpaceScienceQueryOptimizer()
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # Conversation context
        self.conversation_history = []
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize embedding model and vector database."""
        print("Initializing Space Science RAG Assistant...")
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB
        print("Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="space_science_knowledge",
            metadata={"description": "Space science knowledge base for RAG"}
        )
        
        # Load knowledge base if collection is empty
        if self.collection.count() == 0:
            self._load_knowledge_base()
        
        print(f"System initialized with {self.collection.count()} knowledge chunks.")
    
    def _load_knowledge_base(self):
        """Load knowledge base into ChromaDB."""
        print("Loading knowledge base into vector database...")
        
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data['knowledge_base']['chunks']
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['text'])
            
            # Convert metadata to ChromaDB-compatible format
            metadata = chunk['metadata'].copy()
            # Convert list values to comma-separated strings
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ', '.join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
            ids.append(chunk['id'])
        
        # Generate embeddings and add to collection
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"Successfully loaded {len(documents)} chunks into vector database.")
    
    def retrieve_relevant_chunks(self, 
                               query: str, 
                               n_results: int = 5,
                               topic_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """Retrieve relevant knowledge chunks for a query."""
        
        # Optimize query
        optimized_result = self.query_optimizer.optimize_query(query)
        optimized_query = optimized_result.get('optimized_query', query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([optimized_query])[0].tolist()
        
        # Prepare where clause for filtering
        where_clause = None
        if topic_filter:
            where_clause = {"topic": topic_filter}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            chunk = RetrievedChunk(
                id=results['ids'][0][i],
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity_score=1 - results['distances'][0][i]  # Convert distance to similarity
            )
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def generate_response(self, 
                         query: str, 
                         retrieved_chunks: List[RetrievedChunk],
                         include_sources: bool = True) -> Dict[str, Any]:
        """Generate a response based on retrieved chunks."""
        
        # Create context from retrieved chunks
        context_parts = []
        sources = []
        
        for chunk in retrieved_chunks:
            context_parts.append(f"[{chunk.metadata.get('topic', 'Unknown')}] {chunk.text}")
            
            if include_sources:
                source_info = {
                    "id": chunk.id,
                    "topic": chunk.metadata.get('topic', 'Unknown'),
                    "subtopic": chunk.metadata.get('subtopic', 'Unknown'),
                    "source_url": chunk.metadata.get('source', ''),
                    "year": chunk.metadata.get('year', ''),
                    "similarity_score": round(chunk.similarity_score, 3)
                }
                sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Generate response (simplified version - in production, use LLM)
        response = self._create_structured_response(query, context, retrieved_chunks)
        
        return {
            "query": query,
            "response": response,
            "sources": sources,
            "context_used": len(retrieved_chunks),
            "topics_covered": list(set([chunk.metadata.get('topic', 'Unknown') for chunk in retrieved_chunks]))
        }
    
    def _create_structured_response(self, 
                                  query: str, 
                                  context: str, 
                                  chunks: List[RetrievedChunk]) -> str:
        """Create a structured response based on context (simplified version)."""
        
        # Analyze query type
        query_lower = query.lower()
        
        # Get main topics from retrieved chunks
        topics = [chunk.metadata.get('topic', 'Unknown') for chunk in chunks]
        main_topic = max(set(topics), key=topics.count) if topics else "Space Science"
        
        # Create response based on query patterns
        if any(word in query_lower for word in ['what', 'describe', 'explain']):
            response = f"Based on current space science knowledge about {main_topic}:\n\n"
        elif any(word in query_lower for word in ['how', 'why']):
            response = f"Here's how this works in {main_topic}:\n\n"
        elif any(word in query_lower for word in ['when', 'where']):
            response = f"Regarding the location and timing in {main_topic}:\n\n"
        else:
            response = f"Information about {main_topic}:\n\n"
        
        # Add key information from chunks
        key_facts = []
        for chunk in chunks[:3]:  # Use top 3 most relevant chunks
            # Extract key information
            text = chunk.text
            if len(text) > 200:
                text = text[:200] + "..."
            key_facts.append(f"• {text}")
        
        response += "\n".join(key_facts)
        
        # Add related topics suggestion
        unique_topics = list(set([chunk.metadata.get('topic') for chunk in chunks]))
        if len(unique_topics) > 1:
            response += f"\n\nRelated topics you might be interested in: {', '.join(unique_topics[1:3])}"
        
        return response
    
    def ask(self, 
           query: str, 
           n_results: int = 5,
           topic_filter: Optional[str] = None,
           include_sources: bool = True) -> Dict[str, Any]:
        """Main method to ask the assistant a question."""
        
        # Add to conversation history
        self.conversation_history.append({"type": "user", "content": query})
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(
            query, n_results=n_results, topic_filter=topic_filter
        )
        
        # Generate response
        response_data = self.generate_response(
            query, retrieved_chunks, include_sources=include_sources
        )
        
        # Add to conversation history
        self.conversation_history.append({
            "type": "assistant", 
            "content": response_data["response"]
        })
        
        return response_data
    
    def ask_with_history(self, question: str) -> dict:
        """
        Ask a question with enhanced conversation history tracking.
        
        Args:
            question: The user's question
            
        Returns:
            dict: Response with answer, sources, and metadata
        """
        from datetime import datetime
        
        # Add to conversation history with timestamp
        self.conversation_history.append({
            'type': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Use existing ask method
        response_data = self.ask(question)
        
        # Update last assistant entry with timestamp and sources
        if self.conversation_history and self.conversation_history[-1]['type'] == 'assistant':
            self.conversation_history[-1].update({
                'timestamp': datetime.now().isoformat(),
                'sources': response_data.get('sources', [])
            })
        
        return response_data
    
    def get_available_topics(self) -> List[str]:
        """Get list of available topics in the knowledge base."""
        # Query all documents to get unique topics
        all_results = self.collection.get()
        topics = set()
        
        for metadata in all_results['metadatas']:
            if 'topic' in metadata:
                topics.add(metadata['topic'])
        
        return sorted(list(topics))
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base by specific topic."""
        results = self.collection.get(
            where={"topic": topic},
            limit=limit
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        
        return chunks
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

# Example usage and testing
if __name__ == "__main__":
    # Initialize the RAG assistant
    assistant = SpaceScienceRAGAssistant()
    
    # Example queries
    example_queries = [
        "What did the Perseverance rover discover on Mars?",
        "How do black holes form?",
        "Tell me about the James Webb Space Telescope",
        "What is the atmosphere of Mars made of?",
        "How do gravitational waves work?"
    ]
    
    print("\n" + "="*60)
    print("Space Science RAG Assistant - Demo")
    print("="*60)
    
    # Show available topics
    topics = assistant.get_available_topics()
    print(f"\nAvailable topics: {', '.join(topics)}")
    
    # Test example queries
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        response_data = assistant.ask(query)
        
        print(f"Response: {response_data['response'][:200]}...")
        print(f"Topics covered: {', '.join(response_data['topics_covered'])}")
        print(f"Sources used: {response_data['context_used']}")
        
        if response_data['sources']:
            print(f"Top source: {response_data['sources'][0]['topic']} - {response_data['sources'][0]['subtopic']}")
    
    print("\n" + "="*60)
    print("RAG Assistant initialized successfully!")
    print("Use assistant.ask('your question') to interact with it.")
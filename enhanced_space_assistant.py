#!/usr/bin/env python3
"""
Enhanced Space Science RAG Assistant with GPT-4 Integration
Combines ChromaDB retrieval with GPT-4 for advanced response generation.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from query_optimizer import SpaceScienceQueryOptimizer

class EnhancedSpaceScienceAssistant:
    """Enhanced RAG Assistant with GPT-4 integration for space science queries."""
    
    def __init__(self, 
                 knowledge_base_path: str = "space_science_knowledge_base.json",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_api_key: Optional[str] = None,
                 chroma_persist_dir: str = "./enhanced_chroma_db"):
        """
        Initialize the enhanced assistant.
        
        Args:
            knowledge_base_path: Path to the JSON knowledge base
            embedding_model: SentenceTransformer model name
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        print("Initializing Enhanced Space Science Assistant with GPT-4...")
        
        # Initialize OpenAI
        self.openai_client = self._initialize_openai(openai_api_key)
        
        # Initialize components
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Loading embedding model: {embedding_model}")
        
        self.query_optimizer = SpaceScienceQueryOptimizer()
        self.conversation_history = []
        
        # Initialize ChromaDB
        print("Setting up Enhanced ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = "enhanced_space_science_kb"
        self.collection = None
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self._initialize_vector_db()
        
        print(f"Enhanced system initialized with {len(self.knowledge_base)} knowledge chunks.")
    
    def initialize(self):
        """Initialize the assistant (for compatibility with main_assistant.py)."""
        # This method is for compatibility - initialization is done in __init__
        print("Enhanced assistant already initialized.")
        return True
    
    def _initialize_openai(self, api_key: Optional[str]) -> openai.OpenAI:
        """Initialize OpenAI client."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            print("   The assistant will work with basic response generation.")
            return None
        
        try:
            client = openai.OpenAI()
            # Test the connection
            client.models.list()
            print("✅ OpenAI GPT-4 connection established")
            return client
        except Exception as e:
            print(f"❌ Error connecting to OpenAI: {e}")
            return None
    
    def _load_knowledge_base(self, file_path: str) -> List[Dict]:
        """Load knowledge base from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('knowledge_base', {}).get('chunks', [])
        except FileNotFoundError:
            print(f"❌ Knowledge base file not found: {file_path}")
            return []
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return []
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB collection with knowledge base."""
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            # Create new collection
            print("Creating new enhanced collection...")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Enhanced Space Science Knowledge Base with GPT-4"}
            )
            self._populate_vector_db()
    
    def _populate_vector_db(self):
        """Populate ChromaDB with knowledge chunks."""
        if not self.knowledge_base:
            print("❌ No knowledge base to populate")
            return
        
        print(f"Populating vector database with {len(self.knowledge_base)} chunks...")
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in self.knowledge_base:
            documents.append(chunk['text'])
            ids.append(chunk['id'])
            
            # Prepare metadata (convert lists to strings for ChromaDB)
            metadata = chunk['metadata'].copy()
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ', '.join(map(str, value))
                elif not isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"✅ Vector database populated with {len(documents)} chunks")
    
    def rebuild_knowledge_base(self, knowledge_base_path: str = "space_science_knowledge_base.json"):
        """Rebuild the knowledge base by deleting and recreating the collection."""
        try:
            # Delete existing collection
            self.chroma_client.delete_collection(name=self.collection_name)
            print("🔄 Deleted existing collection")
            
            # Reload knowledge base
            self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Enhanced Space Science Knowledge Base with GPT-4"}
            )
            
            # Populate with updated data
            self._populate_vector_db()
            print(f"✅ Knowledge base rebuilt with {len(self.knowledge_base)} chunks")
            
        except Exception as e:
            print(f"❌ Error rebuilding knowledge base: {e}")
    
    def retrieve_knowledge(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge chunks using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            chunks = []
            for i in range(len(results['ids'][0])):
                chunks.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return []
    
    def generate_gpt4_response(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate response using GPT-4 with retrieved context."""
        if not self.openai_client:
            return self._generate_basic_response(question, context_chunks)
        
        try:
            # Prepare context from retrieved chunks
            context_text = "\n\n".join([
                f"Source {i+1} ({chunk['metadata'].get('topic', 'Unknown')}):\n{chunk['text']}"
                for i, chunk in enumerate(context_chunks[:3])
            ])
            
            # Create system prompt
            system_prompt = """
You are an expert space science assistant. Use the provided context to answer questions accurately and comprehensively.

Guidelines:
1. Base your answers primarily on the provided context
2. If the context doesn't fully answer the question, acknowledge this
3. Provide specific details and examples when available
4. Maintain scientific accuracy
5. Be engaging and educational
6. Always cite which sources you're using (Source 1, Source 2, etc.)
"""
            
            # Create user prompt
            user_prompt = f"""
Context Information:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above.
"""
            
            # Call GPT-4
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            gpt_response = response.choices[0].message.content
            
            # Extract topics and sources
            topics_covered = list(set([
                chunk['metadata'].get('topic', 'Unknown')
                for chunk in context_chunks
            ]))
            
            sources = [{
                'topic': chunk['metadata'].get('topic', 'Unknown'),
                'subtopic': chunk['metadata'].get('subtopic', 'Unknown'),
                'year': chunk['metadata'].get('year', 'Unknown'),
                'source_url': chunk['metadata'].get('source', 'Unknown')
            } for chunk in context_chunks]
            
            return {
                'response': gpt_response,
                'sources': sources,
                'topics_covered': topics_covered,
                'model_used': 'GPT-4',
                'chunks_used': len(context_chunks)
            }
            
        except Exception as e:
            print(f"Error with GPT-4 generation: {e}")
            return self._generate_basic_response(question, context_chunks)
    
    def _generate_basic_response(self, question: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback response generation without GPT-4."""
        if not context_chunks:
            return {
                'response': "I don't have enough information to answer that question. Please try rephrasing or asking about a different space science topic.",
                'sources': [],
                'topics_covered': [],
                'model_used': 'Basic',
                'chunks_used': 0
            }
        
        # Simple response combining top chunks
        response_parts = []
        topics_covered = set()
        
        for i, chunk in enumerate(context_chunks[:3]):
            topic = chunk['metadata'].get('topic', 'Space Science')
            topics_covered.add(topic)
            response_parts.append(f"• {chunk['text']}")
        
        response = f"Based on current space science knowledge:\n\n" + "\n\n".join(response_parts)
        
        sources = [{
            'topic': chunk['metadata'].get('topic', 'Unknown'),
            'subtopic': chunk['metadata'].get('subtopic', 'Unknown'),
            'year': chunk['metadata'].get('year', 'Unknown'),
            'source_url': chunk['metadata'].get('source', 'Unknown')
        } for chunk in context_chunks]
        
        return {
            'response': response,
            'sources': sources,
            'topics_covered': list(topics_covered),
            'model_used': 'Basic',
            'chunks_used': len(context_chunks)
        }
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an enhanced GPT-4 response."""
        # Add to conversation history
        self.conversation_history.append({
            'type': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Optimize query
        optimized_query = self.query_optimizer.optimize_query(question)
        
        # Retrieve relevant knowledge
        relevant_chunks = self.retrieve_knowledge(optimized_query['optimized_query'])
        
        # Generate enhanced response
        response_data = self.generate_gpt4_response(question, relevant_chunks)
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'assistant',
            'content': response_data['response'],
            'timestamp': datetime.now().isoformat(),
            'sources': response_data['sources'],
            'model_used': response_data['model_used']
        })
        
        return response_data
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Compatibility method for main_assistant.py - calls ask method."""
        return self.ask(question)
    
    def get_available_topics(self) -> List[str]:
        """Get list of available topics."""
        try:
            results = self.collection.get()
            topics = set()
            for metadata in results['metadatas']:
                if 'topic' in metadata:
                    topics.add(metadata['topic'])
            return sorted(list(topics))
        except Exception as e:
            print(f"Error getting topics: {e}")
            return []
    
    def search_by_topic(self, topic: str, limit: int = 5) -> List[Dict]:
        """Search for chunks by specific topic."""
        try:
            results = self.collection.get(
                where={"topic": {"$eq": topic}},
                limit=limit
            )
            
            chunks = []
            for i in range(len(results['ids'])):
                chunks.append({
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            return chunks
        except Exception as e:
            print(f"Error searching by topic: {e}")
            return []
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced assistant
    assistant = EnhancedSpaceScienceAssistant()
    
    # Test questions
    test_questions = [
        "What did the Perseverance rover discover on Mars?",
        "How do black holes form and what happens at their event horizon?",
        "Tell me about the James Webb Space Telescope's recent discoveries",
        "What is the composition of Mars' atmosphere and how does it compare to Earth?",
        "Explain gravitational waves and their detection by LIGO"
    ]
    
    print("\n" + "="*60)
    print("🚀 ENHANCED SPACE SCIENCE ASSISTANT - GPT-4 POWERED")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Query: {question}")
        print("-" * 40)
        
        response_data = assistant.ask(question)
        
        print(f"Model: {response_data['model_used']}")
        print(f"Response: {response_data['response'][:200]}...")
        print(f"Topics: {', '.join(response_data['topics_covered'])}")
        print(f"Sources: {len(response_data['sources'])}")
        if response_data['sources']:
            print(f"Top source: {response_data['sources'][0]['topic']} - {response_data['sources'][0]['subtopic']}")
    
    print("\n" + "="*60)
    print("Enhanced Assistant initialized successfully!")
    print("Use assistant.ask('your question') to interact with GPT-4 powered responses.")
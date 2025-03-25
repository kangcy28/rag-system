import time
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.services.openai_service import OpenAIService
from app.database.repository import ChunkRepository, DocumentRepository
from app.models.models import QueryRequest, QueryResponse, RetrievedChunk

class RAGService:
    """Service for RAG (Retrieval-Augmented Generation) operations."""
    
    def __init__(self, db: Session):
        """Initialize the RAG service."""
        self.db = db
        self.openai_service = OpenAIService()
    
    def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a query using the RAG approach.
        
        Args:
            query_request (QueryRequest): The query request object
            
        Returns:
            QueryResponse: The response including generated text and retrieved chunks
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = ChunkRepository.retrieve_chunks_for_query(
            self.db, 
            query_request.query, 
            max_chunks=query_request.max_chunks
        )
        
        # Extract chunk contents for context
        context_texts = [chunk["content"] for chunk in retrieved_chunks]
        
        # Generate response using OpenAI with context
        response_data = self.openai_service.generate_response(
            query_request.query, 
            context=context_texts
        )
        
        # Format retrieved chunks for response
        formatted_chunks = []
        for chunk in retrieved_chunks:
            formatted_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    document_title=chunk.get("document_title"),
                    document_source=chunk.get("document_source"),
                    relevance_score=chunk.get("relevance_score")
                )
            )
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            "model": response_data.get("model"),
            "tokens_used": response_data.get("tokens_used"),
            "chunks_retrieved": len(retrieved_chunks)
        }
        
        # Save query and response to database
        ChunkRepository.save_query(
            self.db,
            query_request.query,
            response_data["response"],
            metadata=metadata
        )
        
        # Create response object
        response = QueryResponse(
            query=query_request.query,
            response=response_data["response"],
            chunks=formatted_chunks if query_request.include_sources else None,
            processing_time=processing_time,
            metadata=metadata
        )
        
        return response
    
    def chunk_document(self, document_id: int, chunk_size: int = 200, overlap: int = 50) -> int:
        """
        Chunk a document and store the chunks in the database.
        
        Args:
            document_id (int): The document ID to chunk
            chunk_size (int): The size of each chunk in characters
            overlap (int): The overlap between chunks in characters
            
        Returns:
            int: The number of chunks created
        """
        # Get the document
        document = DocumentRepository.get_document_by_id(self.db, document_id)
        if not document:
            return 0
        
        # Simple text chunking (in a real implementation, this would be more sophisticated)
        text = document["content"]
        chunks = []
        
        # Create chunks with overlap
        for i in range(0, len(text), chunk_size - overlap):
            if i > 0:  # Skip the first overlap
                start = i
            else:
                start = 0
                
            end = min(i + chunk_size, len(text))
            chunk_text = text[start:end]
            
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    "document_id": document_id,
                    "content": chunk_text,
                    "chunk_order": len(chunks) + 1
                })
        
        # Insert chunks into database
        from app.models.models import Chunk
        chunk_objects = [
            Chunk(
                document_id=chunk["document_id"],
                content=chunk["content"],
                chunk_order=chunk["chunk_order"]
            ) 
            for chunk in chunks
        ]
        
        return ChunkRepository.create_chunks_batch(self.db, chunk_objects)
from typing import List, Dict, Any, Optional
import numpy as np
import json
from sqlalchemy.orm import Session

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document as LCDocument

from app.config import get_settings
from app.database.repository import ChunkRepository, DocumentRepository
from app.models.models import Document, Chunk

settings = get_settings()

class LangChainService:
    """Service for LangChain integration with existing database."""
    
    def __init__(self, db: Session):
        """Initialize the LangChain service."""
        self.db = db
        self.embeddings = self._get_embeddings_model()
        self.llm = self._get_llm_model()
        
    def _get_embeddings_model(self):
        """Get the embeddings model."""
        if settings.use_azure:
            return AzureOpenAIEmbeddings(
                azure_endpoint="https://bpragtest.openai.azure.com/openai/deployments/gpt-4o-mini",
                api_key=settings.openai_api_key,
                api_version=settings.api_version,
                deployment=settings.deployment_name,  # Using the same deployment for embeddings
            )
        else:
            # For OpenAI integration if needed in the future
            raise NotImplementedError("Only Azure OpenAI is supported at this time")
    
    def _get_llm_model(self):
        """Get the LLM model."""
        if settings.use_azure:
            return AzureChatOpenAI(
                azure_endpoint="https://bpragtest.openai.azure.com/openai/deployments/gpt-4o-mini",
                api_key=settings.openai_api_key,
                api_version=settings.api_version,
                azure_deployment=settings.deployment_name,
                temperature=0.7,
                max_tokens=5000
            )
        else:
            # For OpenAI integration if needed in the future
            raise NotImplementedError("Only Azure OpenAI is supported at this time")
    
    def process_document(self, document_id: int, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        """
        Process a document using LangChain text splitters and store chunks.
        
        Args:
            document_id (int): The document ID to process
            chunk_size (int): The size of each chunk in characters
            chunk_overlap (int): The overlap between chunks in characters
            
        Returns:
            int: The number of chunks created
        """
        # Get the document
        document = DocumentRepository.get_document_by_id(self.db, document_id)
        if not document:
            return 0
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Split the document text
        splits = text_splitter.split_text(document["content"])
        
        # Create chunk objects
        chunk_objects = []
        for i, split_text in enumerate(splits):
            chunk = Chunk(
                document_id=document_id,
                content=split_text,
                chunk_order=i + 1
            )
            chunk_objects.append(chunk)
        
        # Calculate embeddings for chunks
        # Note: In this implementation, we're not storing the embeddings in the database
        # but we will calculate them on-the-fly for retrieval
        
        # Store chunks in database
        return ChunkRepository.create_chunks_batch(self.db, chunk_objects)
    
    def retrieve_chunks(self, query: str, max_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query using embedding similarity.
        
        Args:
            query (str): The query to find relevant chunks for
            max_chunks (int): The maximum number of chunks to return
            
        Returns:
            List[Dict[str, Any]]: List of retrieved chunks with metadata
        """
        # Get all chunks from the database
        # In a production environment with many documents, you would implement
        # a more efficient retrieval strategy
        query_text = f"""
        SELECT 
            c.chunk_id, c.document_id, c.content, c.chunk_order,
            d.title as document_title, d.source as document_source
        FROM Chunks c
        JOIN Documents d ON c.document_id = d.document_id
        """
        
        result = self.db.execute(query_text)
        chunks = [dict(row._mapping) for row in result]
        
        if not chunks:
            return []
        
        # Calculate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate embeddings for all chunks
        chunk_texts = [chunk["content"] for chunk in chunks]
        chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
        
        # Calculate cosine similarity
        similarities = []
        for i, embedding in enumerate(chunk_embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            chunks[i]["relevance_score"] = float(similarity)
            similarities.append((i, similarity))
        
        # Sort by similarity and take top chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:max_chunks]]
        
        # Return top chunks
        return [chunks[idx] for idx in top_indices]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response using LangChain with retrieved context.
        
        Args:
            query (str): The user's query
            context_chunks (List[Dict[str, Any]]): Retrieved context chunks
            
        Returns:
            Dict[str, Any]: Response with generated text and metadata
        """
        # Format context for the prompt
        context_texts = [chunk["content"] for chunk in context_chunks]
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context_texts)])
        
        # Create the prompt template
        template = """You are a helpful AI assistant. Answer the following question based on the provided context.
If the context doesn't contain relevant information, just say you don't know but provide general information if possible.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        chain = {
            "context": lambda x: context_str,
            "question": lambda x: x
        } | prompt | self.llm
        
        try:
            # Invoke the chain
            response = chain.invoke(query)
            
            # Return formatted response
            return {
                "response": response.content,
                "model": settings.model_name,
                "success": True
            }
        except Exception as e:
            # Handle errors
            return {
                "response": f"Error generating response: {str(e)}",
                "model": settings.model_name,
                "success": False
            }
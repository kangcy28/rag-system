import pandas as pd
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import re

from app.models.models import Document, Chunk, QueryRequest, QueryResponse, RetrievedChunk


class DocumentRepository:
    """Repository for document operations."""
    
    @staticmethod
    def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all documents."""
        query = text("""
            SELECT 
                document_id, title, content, source, 
                created_at, updated_at, document_type
            FROM Documents
            ORDER BY created_at DESC
            OFFSET :skip ROWS
            FETCH NEXT :limit ROWS ONLY
        """)
        result = db.execute(query, {"skip": skip, "limit": limit})
        documents = [dict(row._mapping) for row in result]
        return documents

    @staticmethod
    def get_document_by_id(db: Session, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        query = text("""
            SELECT 
                document_id, title, content, source, 
                created_at, updated_at, document_type
            FROM Documents
            WHERE document_id = :document_id
        """)
        result = db.execute(query, {"document_id": document_id}).first()
        if result:
            return dict(result._mapping)
        return None

    @staticmethod
    def create_document(db: Session, document: Document) -> Dict[str, Any]:
        """Create a new document."""
        query = text("""
            INSERT INTO Documents (title, content, source, document_type)
            OUTPUT INSERTED.*
            VALUES (:title, :content, :source, :document_type)
        """)
        
        result = db.execute(
            query, 
            {
                "title": document.title,
                "content": document.content,
                "source": document.source,
                "document_type": document.document_type
            }
        ).first()
        
        db.commit()
        return dict(result._mapping)

    @staticmethod
    def update_document(db: Session, document_id: int, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing document."""
        # Build dynamic update query
        set_clauses = []
        params = {"document_id": document_id}
        
        for key, value in document.items():
            if value is not None and key in ["title", "content", "source", "document_type"]:
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
        
        if not set_clauses:
            return None
        
        set_clauses.append("updated_at = GETDATE()")
        set_clause = ", ".join(set_clauses)
        
        query = text(f"""
            UPDATE Documents
            SET {set_clause}
            OUTPUT INSERTED.*
            WHERE document_id = :document_id
        """)
        
        result = db.execute(query, params).first()
        db.commit()
        
        if result:
            return dict(result._mapping)
        return None

    @staticmethod
    def delete_document(db: Session, document_id: int) -> bool:
        """Delete a document and all its chunks."""
        # First delete all chunks for this document
        chunk_query = text("""
            DELETE FROM Chunks
            WHERE document_id = :document_id
        """)
        db.execute(chunk_query, {"document_id": document_id})
        
        # Then delete the document
        doc_query = text("""
            DELETE FROM Documents
            OUTPUT DELETED.document_id
            WHERE document_id = :document_id
        """)
        result = db.execute(doc_query, {"document_id": document_id}).first()
        db.commit()
        
        return result is not None


class ChunkRepository:
    """Repository for chunk operations."""
    
    @staticmethod
    def get_chunks_by_document(db: Session, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        query = text("""
            SELECT chunk_id, document_id, content, chunk_order, created_at
            FROM Chunks
            WHERE document_id = :document_id
            ORDER BY chunk_order
        """)
        result = db.execute(query, {"document_id": document_id})
        chunks = [dict(row._mapping) for row in result]
        return chunks

    @staticmethod
    def create_chunk(db: Session, chunk: Chunk) -> Dict[str, Any]:
        """Create a new chunk."""
        query = text("""
            INSERT INTO Chunks (document_id, content, chunk_order)
            OUTPUT INSERTED.*
            VALUES (:document_id, :content, :chunk_order)
        """)
        
        result = db.execute(
            query, 
            {
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_order": chunk.chunk_order
            }
        ).first()
        
        db.commit()
        return dict(result._mapping)

    @staticmethod
    def create_chunks_batch(db: Session, chunks: List[Chunk]) -> int:
        """Create multiple chunks in a batch."""
        if not chunks:
            return 0
        
        # Using executemany with PyODBC
        conn = db.connection().connection
        cursor = conn.cursor()
        
        sql = """
            INSERT INTO Chunks (document_id, content, chunk_order)
            VALUES (?, ?, ?)
        """
        
        data = [(c.document_id, c.content, c.chunk_order) for c in chunks]
        cursor.executemany(sql, data)
        
        db.commit()
        return len(chunks)

    @staticmethod
    def retrieve_chunks_for_query(db: Session, query_text: str, max_chunks: int = 5) -> List[Dict[str, Any]]:
        """
        Simple chunk retrieval based on keyword matching.
        This replaces the previous embedding-based retrieval.
        """
        # Extract keywords from the query
        keywords = ChunkRepository._extract_keywords(query_text)
        
        # Build a dynamic SQL query that searches for any of the keywords
        where_clauses = []
        params = {"max_chunks": max_chunks}
        
        for i, keyword in enumerate(keywords):
            param_name = f"keyword_{i}"
            where_clauses.append(f"c.content LIKE :{param_name}")
            params[param_name] = f"%{keyword}%"
        
        where_clause = " OR ".join(where_clauses) if where_clauses else "1=1"
        
        query = text(f"""
            SELECT TOP :max_chunks
                c.chunk_id, c.document_id, c.content, c.chunk_order,
                d.title as document_title, d.source as document_source
            FROM Chunks c
            JOIN Documents d ON c.document_id = d.document_id
            WHERE {where_clause}
            ORDER BY c.chunk_id
        """)
        
        result = db.execute(query, params)
        
        chunks = []
        for row in result:
            row_dict = dict(row._mapping)
            # Calculate a relevance score based on keyword matches
            relevance_score = ChunkRepository._calculate_relevance_score(row_dict["content"], keywords)
            row_dict["relevance_score"] = relevance_score
            chunks.append(row_dict)
        
        # Sort by relevance score
        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return chunks[:max_chunks]
    
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract keywords from text."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words
        words = text.split()
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "is", "are", "was", "were"}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    @staticmethod
    def _calculate_relevance_score(text: str, keywords: List[str]) -> float:
        """Calculate a relevance score based on keyword matches."""
        text = text.lower()
        score = 0.0
        
        for keyword in keywords:
            if keyword in text:
                # Add 1 point for each keyword found
                score += 1.0
                
                # Add extra points for multiple occurrences
                occurrences = text.count(keyword)
                if occurrences > 1:
                    score += 0.2 * (occurrences - 1)
        
        return score

    @staticmethod
    def save_query(db: Session, query_text: str, response_text: str, metadata: Dict = None) -> Dict[str, Any]:
        """Save query and response to the database."""
        metadata_json = json.dumps(metadata) if metadata else None
        
        query = text("""
            INSERT INTO Queries (query_text, response_text, metadata)
            OUTPUT INSERTED.*
            VALUES (:query_text, :response_text, :metadata)
        """)
        
        result = db.execute(
            query, 
            {
                "query_text": query_text,
                "response_text": response_text,
                "metadata": metadata_json
            }
        ).first()
        
        db.commit()
        return dict(result._mapping)
USE RagDatabase;
GO

-- Insert sample documents for testing
IF NOT EXISTS (SELECT TOP 1 * FROM Documents)
BEGIN
    INSERT INTO Documents (title, content, source, document_type)
    VALUES 
    ('Introduction to RAG Systems', 'Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based and generation-based approaches. It enhances large language models by retrieving relevant information from external knowledge sources.', 'Technical Documentation', 'article'),
    ('LangChain Documentation', 'LangChain is a framework for developing applications powered by language models. It provides tools and abstractions for working with LLMs, including RAG systems.', 'API Documentation', 'documentation'),
    ('MSSQL Integration Guide', 'This guide explains how to integrate Microsoft SQL Server with Python applications. Topics include connection management, query optimization, and containerization with Docker.', 'Technical Guide', 'guide');

    -- Create chunks for each document
    -- Document 1 chunks
    INSERT INTO Chunks (document_id, content, chunk_order)
    VALUES 
    (1, 'Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based and generation-based approaches.', 1),
    (1, 'It enhances large language models by retrieving relevant information from external knowledge sources.', 2);

    -- Document 2 chunks
    INSERT INTO Chunks (document_id, content, chunk_order)
    VALUES 
    (2, 'LangChain is a framework for developing applications powered by language models.', 1),
    (2, 'It provides tools and abstractions for working with LLMs, including RAG systems.', 2);

    -- Document 3 chunks
    INSERT INTO Chunks (document_id, content, chunk_order)
    VALUES 
    (3, 'This guide explains how to integrate Microsoft SQL Server with Python applications.', 1),
    (3, 'Topics include connection management, query optimization, and containerization with Docker.', 2);

    PRINT 'Sample data inserted successfully.';
END
ELSE
BEGIN
    PRINT 'Sample data already exists.';
END
GO
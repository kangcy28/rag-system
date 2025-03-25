-- Create the RAG database if it doesn't exist
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'RagDatabase')
BEGIN
    CREATE DATABASE RagDatabase;
END
GO

USE RagDatabase;
GO

-- Create tables
-- Documents table to store all document content
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Documents')
BEGIN
    CREATE TABLE Documents (
        document_id INT IDENTITY(1,1) PRIMARY KEY,
        title NVARCHAR(255) NOT NULL,
        content NVARCHAR(MAX) NOT NULL,
        source NVARCHAR(255),
        created_at DATETIME DEFAULT GETDATE(),
        updated_at DATETIME DEFAULT GETDATE(),
        document_type NVARCHAR(50)
    );
END
GO

-- Chunks table to store text chunks for retrieval
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Chunks')
BEGIN
    CREATE TABLE Chunks (
        chunk_id INT IDENTITY(1,1) PRIMARY KEY,
        document_id INT FOREIGN KEY REFERENCES Documents(document_id),
        content NVARCHAR(MAX) NOT NULL,
        chunk_order INT NOT NULL,
        created_at DATETIME DEFAULT GETDATE()
    );
END
GO

-- Queries table to store user queries and responses
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Queries')
BEGIN
    CREATE TABLE Queries (
        query_id INT IDENTITY(1,1) PRIMARY KEY,
        query_text NVARCHAR(1000) NOT NULL,
        response_text NVARCHAR(MAX),
        created_at DATETIME DEFAULT GETDATE(),
        metadata NVARCHAR(MAX) -- JSON field to store relevant chunks, etc.
    );
END
GO

-- Create indexes for better performance
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_Chunks_DocumentId' AND object_id = OBJECT_ID('Chunks'))
BEGIN
    CREATE INDEX IX_Chunks_DocumentId ON Chunks (document_id);
END
GO


-- Run additional setup scripts
:r /usr/config/setup/create_tables.sql
GO

PRINT 'Database initialization completed successfully.';
GO
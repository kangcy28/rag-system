# RAG Application with FastAPI and Azure OpenAI

A Retrieval-Augmented Generation (RAG) application using FastAPI, SQLAlchemy, and Azure OpenAI for enhanced question answering based on document context. Both backend and database are fully containerized with Docker for easy deployment.

## Overview

This application provides:

- A RAG system that retrieves document chunks from a database to enhance AI responses
- REST API for document management (upload, update, delete)
- Document chunking for efficient retrieval
- Question answering with context from stored documents
- Azure OpenAI integration for content generation

## Architecture

The application uses a three-tier architecture:

- **Backend**: FastAPI Python application
- **Database**: Microsoft SQL Server
- **Frontend**: (Placeholder for a Vue.js frontend - to be implemented)

## Technologies

- **Backend**:
  - FastAPI (web framework)
  - SQLAlchemy (ORM with PyMSSQL)
  - LangChain (for chunking and RAG processing)
  - Azure OpenAI integration
  - Docker for containerization

- **Database**:
  - Microsoft SQL Server 2022
  - Containerized using Docker

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── config.py              # Application configuration
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── database/              # Database connection and repositories
│   │   ├── models/                # Pydantic models
│   │   ├── routes/                # API routes
│   │   └── services/              # Business logic services
│   ├── Dockerfile                 # Docker configuration for backend
│   └── requirements.txt           # Python dependencies
├── database/
│   ├── Dockerfile                 # SQL Server Docker configuration
│   ├── init.sql                   # Database initialization script
│   └── setup/                     # Additional database setup scripts
├── docker-compose.yml             # Docker Compose configuration
└── .gitignore                     # Git ignore file
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Azure OpenAI API key

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# Database configuration
DB_USER=sa
DB_PASSWORD=YourStrongPassword!
DB_NAME=RagDatabase

# OpenAI API configuration
OPENAI_API_KEY=your-openai-api-key

# Azure OpenAI configuration
USE_AZURE=True
AZURE_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_ENDPOINT=your-azure-endpoint
AZURE_API_VERSION=2024-12-01-preview
```

### Starting the Application

1. Build and start the containerized services:

```bash
docker-compose up -d
```

This command will build and start both the FastAPI backend and SQL Server database containers.

2. The API will be available at http://localhost:8000
3. API documentation is available at http://localhost:8000/docs

## API Endpoints

### RAG Operations

- `POST /api/query`: Process a query using the RAG system

### Document Management

- `GET /api/documents`: Get all documents with pagination
- `GET /api/documents/{document_id}`: Get a specific document
- `POST /api/documents`: Create a new document
- `PUT /api/documents/{document_id}`: Update a document
- `DELETE /api/documents/{document_id}`: Delete a document
- `POST /api/documents/{document_id}/process`: Process a document to create chunks

### Chunk Management

- `GET /api/documents/{document_id}/chunks`: Get all chunks for a document
- `POST /api/chunks`: Create a new chunk

## Database Schema

The database consists of three main tables:

1. **Documents**: Stores document metadata and content
2. **Chunks**: Stores text chunks created from documents
3. **Queries**: Records user queries and system responses

## How RAG Works in This Application

1. **Document Processing**:
   - Documents are stored in the database
   - LangChain's RecursiveCharacterTextSplitter divides documents into chunks
   - Chunks are stored in the database for retrieval

2. **Query Processing**:
   - User submits a query
   - The system retrieves relevant chunks using embedding similarity 
   - Azure OpenAI generates a response using the query and retrieved context
   - Both query and response are stored for future reference

## Development

The application is fully containerized, but you can also set up a local development environment if needed.

### Local Development Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Run the backend with hot reload:

```bash
cd backend
uvicorn app.main:app --reload
```

### Adding Documents

Documents can be added via the API. After adding a document, use the `/api/documents/{document_id}/process` endpoint to chunk the document for retrieval.

## Future Improvements

- Add user authentication
- Build a Vue.js frontend interface
- Add document upload via file upload (PDF, DOCX)
- Implement vector database for more efficient embeddings storage
- Add batch processing for large documents
- Improve chunking strategies
- Add unit and integration tests

## License

[MIT License](LICENSE)

import pymssql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# Create a connection string for pymssql that works with SQLAlchemy
# Format: "mssql+pymssql://username:password@server:port/database"
connection_url = f"mssql+pymssql://{settings.db_user}:{settings.db_password}@{settings.db_server}:{settings.db_port}/{settings.db_name}"

# Create SQLAlchemy engine with PyMSSQL
engine = create_engine(
    connection_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_pymssql_connection():
    """
    Get direct PyMSSQL connection for complex queries.
    
    Returns:
        Connection: PyMSSQL database connection.
    """
    return pymssql.connect(
        server=settings.db_server,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name
    )
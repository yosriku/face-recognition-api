import os
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
import numpy as np

# Load database connection string from environment variable for security and configurability
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/face_db")

# Initialize the SQLAlchemy database engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class and a session instance
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define the base class for ORM models
Base = declarative_base()

# Define the Face model to store identity name and corresponding face embedding
class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)  # Unique identifier for each face entry
    name = Column(String, index=True)  # Name associated with the face embedding
    embedding = Column(LargeBinary, nullable=False)  # Serialized face embedding stored as binary

# Automatically create the 'faces' table in the database if it doesn't already exist
Base.metadata.create_all(bind=engine)

# Helper function to serialize a NumPy embedding array into bytes for database storage
def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return embedding.tobytes()

# Helper function to deserialize binary data back into a NumPy embedding array
def bytes_to_embedding(embedding_bytes: bytes) -> np.ndarray:
    return np.frombuffer(embedding_bytes, dtype=np.float32)

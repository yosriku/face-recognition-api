import os
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base
import numpy as np

# Use environment variables for database credentials
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/face_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    # Store the embedding as raw bytes
    embedding = Column(LargeBinary, nullable=False)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

# Helper functions to convert between numpy array and bytes
def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return embedding.tobytes()

def bytes_to_embedding(embedding_bytes: bytes) -> np.ndarray:
    return np.frombuffer(embedding_bytes, dtype=np.float32)
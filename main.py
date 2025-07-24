from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

# Import local modules for database and face processing logic
from database import SessionLocal, Face, embedding_to_bytes, bytes_to_embedding
from face_processor import detect_face, extract_features, match_faces, MATCH_THRESHOLD
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(title="Face Recognition API")

# Request model for face registration
class FaceRegisterRequest(BaseModel):
    name: str

# Response model for face data
class FaceResponse(BaseModel):
    id: int
    name: str

# Dependency to manage database session lifecycle
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/face/register", response_model=FaceResponse, status_code=201)
async def register_face(name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint to register a new face.
    Accepts a name and an image file, detects the face, extracts features,
    and stores the embedding in the database.
    """
    image_bytes = await file.read()

    # Perform face detection
    cropped_face = detect_face(image_bytes)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Extract embedding and convert to bytes
    embedding = extract_features(cropped_face)
    embedding_bytes = embedding_to_bytes(embedding)

    # Store face record in the database
    db_face = Face(name=name, embedding=embedding_bytes)
    db.add(db_face)
    db.commit()
    db.refresh(db_face)

    return db_face

@app.post("/api/face/recognize")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint to recognize a face from an uploaded image.
    Compares the face embedding with known embeddings in the database
    and returns the best match if similarity exceeds the threshold.
    """
    image_bytes = await file.read()
    cropped_face = detect_face(image_bytes)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Extract embedding from input image
    query_embedding = extract_features(cropped_face)

    # Retrieve all registered faces
    known_faces = db.query(Face).all()
    if not known_faces:
        raise HTTPException(status_code=404, detail="No faces registered in the database.")

    # Initialize best match variables
    best_match_name = "Unknown"
    highest_similarity = 0.0

    # Compare input embedding with each stored embedding
    for known_face in known_faces:
        known_embedding = bytes_to_embedding(known_face.embedding)
        similarity = float(match_faces(query_embedding, known_embedding))

        # Update best match if similarity is higher
        if similarity > highest_similarity:
            highest_similarity = similarity
            if similarity > MATCH_THRESHOLD:
                best_match_name = known_face.name

    # Return matched name if threshold passed
    if best_match_name != "Unknown":
        return {
            "matched_name": best_match_name,
            "similarity": round(float(highest_similarity), 4)
        }
    else:
        # No match passed the threshold
        raise HTTPException(
            status_code=404,
            detail=f"No match found. Highest similarity: {round(float(highest_similarity), 4)}"
        )

@app.get("/api/face", response_model=List[FaceResponse])
def get_all_faces(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all registered faces from the database.
    Returns a list of face records (ID and name only).
    """
    faces = db.query(Face).all()
    return faces

@app.delete("/api/face/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db)):
    """
    Endpoint to delete a registered face by its ID.
    """
    face_to_delete = db.query(Face).filter(Face.id == face_id).first()
    if not face_to_delete:
        raise HTTPException(status_code=404, detail=f"Face with id {face_id} not found.")

    db.delete(face_to_delete)
    db.commit()

    return {"status": "success", "message": f"Face with id {face_id} deleted."}

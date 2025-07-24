from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List


# Import your database and face processing modules
from database import SessionLocal, Face, embedding_to_bytes, bytes_to_embedding
from face_processor import detect_face, extract_features, match_faces, MATCH_THRESHOLD
from pydantic import BaseModel

app = FastAPI(title="Face Recognition API")

# Pydantic models for request and response data validation
class FaceRegisterRequest(BaseModel):
    name: str

class FaceResponse(BaseModel):
    id: int
    name: str

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/face/register", response_model=FaceResponse, status_code=201)
async def register_face(name: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Registers a new face with a name."""
    image_bytes = await file.read()
    
    cropped_face = detect_face(image_bytes)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
        
    embedding = extract_features(cropped_face)
    embedding_bytes = embedding_to_bytes(embedding)
    
    db_face = Face(name=name, embedding=embedding_bytes)
    db.add(db_face)
    db.commit()
    db.refresh(db_face)
    
    return db_face

@app.post("/api/face/recognize")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await file.read()
    cropped_face = detect_face(image_bytes)
    if cropped_face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    query_embedding = extract_features(cropped_face)
    known_faces = db.query(Face).all()

    if not known_faces:
        raise HTTPException(status_code=404, detail="No faces registered in the database.")

    best_match_name = "Unknown"
    highest_similarity = 0.0

    for known_face in known_faces:
        known_embedding = bytes_to_embedding(known_face.embedding)
        similarity = float(match_faces(query_embedding, known_embedding))  
        if similarity > highest_similarity:
            highest_similarity = similarity
            if similarity > MATCH_THRESHOLD:
                best_match_name = known_face.name

    if best_match_name != "Unknown":
        return {
            "matched_name": best_match_name,
            "similarity": round(float(highest_similarity), 4)  
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No match found. Highest similarity: {round(float(highest_similarity), 4)}"
        )


@app.get("/api/face", response_model=List[FaceResponse])
def get_all_faces(db: Session = Depends(get_db)):
    """Returns a list of all registered faces."""
    faces = db.query(Face).all()
    return faces

@app.delete("/api/face/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db)):
    """Deletes a face from the database by its ID."""
    face_to_delete = db.query(Face).filter(Face.id == face_id).first()
    
    if not face_to_delete:
        raise HTTPException(status_code=404, detail=f"Face with id {face_id} not found.")
        
    db.delete(face_to_delete)
    db.commit()
    
    return {"status": "success", "message": f"Face with id {face_id} deleted."}
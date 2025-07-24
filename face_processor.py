import cv2
import numpy as np
import os
import time
from typing import Optional

# Import MTCNN for face detection and FaceNet for face embedding extraction
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize the face detector and face recognizer
detector = MTCNN()
embedder = FaceNet()

def align_face(img, landmarks):
    # Resize cropped face to the input size expected by FaceNet
    return cv2.resize(img, (160, 160))

def detect_face(image_bytes: bytes, conf_thresh: float = 0.9) -> Optional[np.ndarray]:
    """
    Detect a face from raw image bytes, return aligned face if confidence is sufficient.
    Saves detection and aligned face images for logging/debugging purposes.
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("❌ Failed to decode image.")
        return None

    # Perform face detection using MTCNN
    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not detections:
        print("❌ No face detected.")
        os.makedirs("detections", exist_ok=True)
        fail_filename = f"detections/no_face_{int(time.time())}.jpg"

        # Annotate and save the failed detection image
        cv2.putText(img, "NO FACE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(fail_filename, img)
        return None

    # Select the face with the highest confidence
    best = max(detections, key=lambda x: x["confidence"])
    best_score = best["confidence"]
    print(f"✅ Best face score: {best_score:.3f}")

    # Reject if below confidence threshold
    if best_score < conf_thresh:
        print(f"⚠️ Score below threshold {conf_thresh}, rejected.")
        return None

    # Get face bounding box and landmarks
    x, y, w, h = best["box"]
    keypoints = best["keypoints"]

    # Annotate the detected face and save detection image
    os.makedirs("detections", exist_ok=True)
    timestamp = int(time.time())
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (kx, ky) in keypoints.values():
        cv2.circle(img, (kx, ky), 2, (0, 0, 255), -1)

    detect_filename = f"detections/detected_{timestamp}.jpg"
    cv2.imwrite(detect_filename, img)
    print(f"💾 Detection saved: {detect_filename}")

    # Crop and align face before feature extraction
    face_crop = img[y:y + h, x:x + w]
    aligned_face = align_face(face_crop, keypoints)

    aligned_filename = f"detections/aligned_{timestamp}.jpg"
    cv2.imwrite(aligned_filename, aligned_face)
    print(f"💾 Aligned face saved: {aligned_filename}")

    return aligned_face

def extract_features(cropped_face: np.ndarray) -> np.ndarray:
    """
    Extract normalized face embedding using FaceNet.
    The input face must be in RGB format and 160x160 resolution.
    """
    face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    embedding = embedder.embeddings([face_resized])[0]

    # Normalize the embedding vector (L2 normalization)
    embedding_norm = embedding / np.linalg.norm(embedding)
    return embedding_norm.flatten()

# Similarity threshold for face verification (can be tuned based on use case)
MATCH_THRESHOLD = 0.5 

def match_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two face embeddings.
    Result is clipped between 0.0 and 1.0 for stability.
    """
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return float(np.clip(similarity, 0.0, 1.0))

# Face Recognition API 

This is an end-to-end face recognition system based on deep learning.  
The system can detect faces in images, extract facial features, and match them against a database of registered faces.

---

## 🚀 Features
- **Face Detection** → Crop and extract face regions from input images.  
- **Feature Extraction** → Generate embeddings (facial feature vectors) from cropped faces.  
- **Face Matching** → Compare input face embeddings with stored embeddings.  
- **Database Management** → Store, retrieve, and delete registered faces.  
- **REST API (FastAPI)** → 4 endpoints available:
  - `GET /api/face` – List all faces in database  
  - `POST /api/face/register` – Register a new face  
  - `POST /api/face/recognize` – Recognize a face  
  - `DELETE /api/face/{id}` – Delete a face by ID

---

## 🛠️ Tech Stack
- **Programming Language**: Python 3.10  
- **Framework**: FastAPI + Uvicorn  
- **Database**: PostgreSQL 13  
- **Containerization**: Docker + Docker Compose  
- **Libraries**: OpenCV, NumPy, SQLAlchemy, MTCNN, Keras

---

## ⚙️ Installation & Running

### **Running with Docker**

#### Clone the repository:
```bash
git clone https://github.com/yosriku/face-recognition-api.git
cd face-recognition-api
```

#### (Optional) Login to docker
```bash
docker login
```

#### Run Docker Compose:
```bash
docker-compose up
```

After startup:
- API: **http://localhost:8000/docs**  
- PostgreSQL: **localhost:5432**  

---

---

## 🔗 API Endpoints

Click the dropdown according to the endpoint function and click Try Out to test the endpoint.

### 1. **Register a Face**
`POST /api/face/register`

**Form Data:**
- `name`: string (person name)
- `file`: image (JPG/PNG)

**Response:**
```json
{
  "id": 1,
  "name": "Yosriko"
}
```

### 2. **Recognize a Face**
`POST /api/face/recognize`

**Form Data:**
- `file`: image (JPG/PNG)

**Response (Match Found):**
```json
{
  "matched_name": "Yosriko",
  "similarity": 0.9231
}
```

**Response (No Match):**
```json
{
  "detail": "No match found. Highest similarity: 0.45"
}
```

### 3. **Get All Faces**
`GET /api/face`

**Response:**
```json
[
  {
    "id": 1,
    "name": "Yosriko"
  },
  {
    "id": 2,
    "name": "Henry Cavill"
  }
]
```

### 4. **Delete a Face**
`DELETE /api/face/{id}`

**Response:**
```json
{
  "status": "success",
  "message": "Face with id 1 deleted."
}
```

---

## 🧪 Example Use Case

1. **Register a new face** → Upload an image of the person with a name.  
2. **Recognize** → Upload another image; the system will compare and return the closest match.  
3. **Get** → Check all registered faces.  
4. **Delete** → Remove a face if needed.

---

## 📂 Project Structure

```
├── database.py          # Database models & connection
├── face_processor.py    # Face detection, extract, and match
├── main.py              # FastAPI routes
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## ✅ Author
Developed by **Yosriko Rahmat Karoni Sabelekake**  
For **Widya Group AI Engineer Knowledge Test**


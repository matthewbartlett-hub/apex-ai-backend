from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # This will confirm the backend received the file
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "File received successfully"
    }

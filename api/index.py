import os
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from google.cloud import vision

app = FastAPI()

# CORS so Streamlit can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Load Google Vision credentials from environment var
# ---------------------------------------------------

cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if cred_json is None:
    raise Exception("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

# Parse JSON key (which you pasted into Vercel)
cred_dict = json.loads(cred_json)

# Build credentials object
credentials = service_account.Credentials.from_service_account_info(cred_dict)

# Create a Vision API client
vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.get("/")
def root():
    return {"message": "Backend with OCR is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Read file bytes
    content = await file.read()

    # Determine if this is a PDF or an image
    is_pdf = file.filename.lower().endswith(".pdf")

    if is_pdf:
        # Google Vision handles PDF OCR using document_text_detection
        image = vision.Image(content=content)

        response = vision_client.document_text_detection(image=image)
        text = response.full_text_annotation.text

    else:
        # Normal image OCR
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)

        # Extract text from detection result
        texts = response.text_annotations
        text = texts[0].description if texts else ""

    # Send OCR text back to Streamlit
    return {
        "filename": file.filename,
        "ocr_text": text,
        "status": "OCR completed"
    }

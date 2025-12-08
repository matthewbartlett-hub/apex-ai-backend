import os
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from google.cloud import vision

app = FastAPI()

# Allow Streamlit frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Vision client (assigned during startup)
vision_client = None


# ---------------------------
# SAFE STARTUP (NO CRASH BEFORE UVICORN STARTS)
# ---------------------------
@app.on_event("startup")
def startup_event():
    global vision_client

    cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not cred_json:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

    cred_dict = json.loads(cred_json)
    credentials = service_account.Credentials.from_service_account_info(cred_dict)

    vision_client = vision.ImageAnnotatorClient(credentials=credentials)


# ---------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------
@app.get("/")
def root():
    return {"status": "Backend running"}


# ---------------------------
# MAIN OCR ENDPOINT
# ---------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()

    # ---------------------------
    # PDF OCR (SYNCHRONOUS BATCH)
    # ---------------------------
    if filename.endswith(".pdf"):
        input_config = vision.InputConfig(
            content=content,
            mime_type="application/pdf"
        )

        feature = vision.Feature(
            type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
        )

        file_request = vision.AnnotateFileRequest(
            input_config=input_config,
            features=[feature]
        )

        batch_request = vision.BatchAnnotateFilesRequest(
            requests=[file_request]
        )

        result = vision_client.batch_annotate_files(request=batch_request)

        full_text = ""
        for response in result.responses:
            if response.full_text_annotation.text:
                full_text += response.full_text_annotation.text + "\n"

        return {
            "filename": filename,
            "ocr_text": full_text.strip(),
            "status": "PDF processed"
        }

    # ---------------------------
    # IMAGE OCR
    # ---------------------------
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)

    text = response.text_annotations[0].description if response.text_annotations else ""

    return {
        "filename": filename,
        "ocr_text": text,
        "status": "Image processed"
    }

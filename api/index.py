import os
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.oauth2 import service_account
from google.cloud import vision

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Google Vision credentials
cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if cred_json is None:
    raise Exception("Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

cred_dict = json.loads(cred_json)
credentials = service_account.Credentials.from_service_account_info(cred_dict)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)


@app.get("/")
def root():
    return {"message": "Backend with OCR is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()

    # ---------------------------
    # PDF HANDLING (correct API)
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

        batch_request = vision.AsyncBatchAnnotateFilesRequest(
            requests=[file_request]
        )

        operation = vision_client.async_batch_annotate_files(requests=[file_request])

        result = operation.result(timeout=180)

        # Extract text from all responses and pages
        full_text = ""
        for response in result.responses:
            if response.full_text_annotation.text:
                full_text += response.full_text_annotation.text + "\n"

        return {
            "filename": file.filename,
            "ocr_text": full_text.strip(),
            "status": "OCR completed (PDF)"
        }

    # ---------------------------
    # IMAGE HANDLING
    # ---------------------------
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)

    texts = response.text_annotations
    text = texts[0].description if texts else ""

    return {
        "filename": file.filename,
        "ocr_text": text,
        "status": "OCR completed (image)"
    }

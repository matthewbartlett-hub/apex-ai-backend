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

    # Use synchronous batch call (correct for this request type)
    batch_request = vision.BatchAnnotateFilesRequest(
        requests=[file_request]
    )

    result = vision_client.batch_annotate_files(request=batch_request)

    full_text = ""
    for response in result.responses:
        if response.full_text_annotation.text:
            full_text += response.full_text_annotation.text + "\n"

    return {
        "filename": file.filename,
        "ocr_text": full_text.strip(),
        "status": "OCR completed (PDF)"
    }

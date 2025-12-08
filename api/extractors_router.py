# api/extractors_router.py

from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.architects_extractor import ArchitectsExtractor, TemplateExtractor


router = APIRouter()


class ExtractionRequest(BaseModel):
    ocr_text: str


class ExtractionResponse(BaseModel):
    template_id: Optional[str]
    profession: Optional[str]
    insurer: Optional[str]
    insurer_confidence: float
    fields_raw: Dict[str, Any]
    fields_normalized: Dict[str, Any]


EXTRACTORS: List[TemplateExtractor] = [
    ArchitectsExtractor(),
]


def choose_best_extractor(text: str) -> Optional[TemplateExtractor]:
    best = None
    best_score = 0.0
    for extractor in EXTRACTORS:
        score = extractor.match_score(text)
        if score > best_score:
            best = extractor
            best_score = score
    if best_score < 0.5:
        return None
    return best


@router.post("/extract", response_model=ExtractionResponse)
async def extract(payload: ExtractionRequest) -> ExtractionResponse:
    text = (payload.ocr_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="ocr_text is required")

    extractor = choose_best_extractor(text)
    if extractor is None:
        raise HTTPException(status_code=422, detail="No suitable extractor found")

    raw, norm = extractor.extract(text)

    return ExtractionResponse(
        template_id=extractor.template_id,
        profession=extractor.profession,
        insurer=None,
        insurer_confidence=0.0,
        fields_raw=raw,
        fields_normalized=norm,
    )

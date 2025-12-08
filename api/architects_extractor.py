# app/architects_extractor.py

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


# =============================
# Pydantic models
# =============================

class ExtractionRequest(BaseModel):
    ocr_text: str


class ExtractionResponse(BaseModel):
    template_id: Optional[str]
    profession: Optional[str]
    insurer: Optional[str]
    insurer_confidence: float
    fields_raw: Dict[str, Any]
    fields_normalized: Dict[str, Any]


# =============================
# Normalisation helpers
# =============================

_money_cleaner = re.compile(r"[^\d\.]")

def parse_money(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    v = value.strip().lower().replace(",", "")
    multiplier = 1.0
    if "k" in v:
        multiplier = 1_000.0
        v = v.replace("k", "")
    if "m" in v:
        multiplier = 1_000_000.0
        v = v.replace("m", "")
    v = v.replace("£", "")
    v = _money_cleaner.sub("", v)
    if not v:
        return None
    try:
        return float(v) * multiplier
    except ValueError:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip()
    fmts = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%Y-%m-%d"
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(v, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    return None


# =============================
# Base extractor interface
# =============================

class TemplateExtractor:
    template_id: str = "base"
    profession: str = "unknown"

    def match_score(self, text: str) -> float:
        """Return 0.0 to 1.0 likelihood that this extractor fits."""
        raise NotImplementedError

    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (raw_fields, normalized_fields)."""
        raise NotImplementedError


# =============================
# Architects extractor
# =============================

class ArchitectsExtractor(TemplateExtractor):
    template_id = "apex_architects_v1"
    profession = "architects"

    # Key labels as they appear in the template
    LABELS = {
        "firm_name": ["Full trading names of all Firms", "Name(s)"],
        "date_established": ["Date Established"],
        "website": ["2a) Website"],
        "email": ["2b) Email Address"],
        "telephone": ["2c) Telephone Number"],
        "total_staff_block": ["Total Number of Staff"],
        "current_pi_block": ["Current Professional Indemnity Policy"],
        "financial_year_end": ["Financial Year End"],
    }

    def match_score(self, text: str) -> float:
        t = text.lower()
        score = 0.0
        if "professional indemnity insurance proposal form for architects" in t:
            score += 0.7
        if "breakdown of your activities and percentage of income generated for each discipline" in t:
            score += 0.2
        if "breakdown of contract types described below and percentage of income generated" in t:
            score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _lines(text: str) -> List[str]:
        return [ln.strip() for ln in text.splitlines()]

    @classmethod
    def _find_line_index(cls, lines: List[str], needles: List[str]) -> Optional[int]:
        lower_lines = [ln.lower() for ln in lines]
        for i, line in enumerate(lower_lines):
            for needle in needles:
                if needle.lower() in line:
                    return i
        return None

    @classmethod
    def _extract_after_label(
        cls,
        lines: List[str],
        labels: List[str],
        max_lookahead: int = 3
    ) -> Optional[str]:
        idx = cls._find_line_index(lines, labels)
        if idx is None:
            return None
        # Try same line after colon
        line = lines[idx]
        parts = re.split(r"[:\-]", line, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        # Else look at next non empty line(s)
        for j in range(1, max_lookahead + 1):
            if idx + j >= len(lines):
                break
            nxt = lines[idx + j].strip()
            if nxt:
                return nxt
        return None

    def _extract_staff_block(self, lines: List[str]) -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {
            "staff_principals_raw": None,
            "staff_qualified_raw": None,
            "staff_unqualified_raw": None,
            "staff_others_raw": None,
        }
        idx = self._find_line_index(lines, self.LABELS["total_staff_block"])
        if idx is None:
            return res
        # In many OCRs the next lines contain the headings then values.
        # We will scan a small window for numeric values.
        window = lines[idx : idx + 6]
        joined = " ".join(window)
        # Very simple patterns, to be refined with real OCR examples
        m_princ = re.search(r"Principals\s+(\d+)", joined, re.IGNORECASE)
        m_q = re.search(r"Qualified Staff\s+(\d+)", joined, re.IGNORECASE)
        m_u = re.search(r"Unqualified Staff\s+(\d+)", joined, re.IGNORECASE)
        m_o = re.search(r"Others\s+(\d+)", joined, re.IGNORECASE)
        if m_princ:
            res["staff_principals_raw"] = m_princ.group(1)
        if m_q:
            res["staff_qualified_raw"] = m_q.group(1)
        if m_u:
            res["staff_unqualified_raw"] = m_u.group(1)
        if m_o:
            res["staff_others_raw"] = m_o.group(1)
        return res

    def _extract_current_pi_block(self, text: str) -> Dict[str, Optional[str]]:
        res = {
            "current_pi_insurer_raw": None,
            "current_pi_broker_raw": None,
            "current_pi_limit_raw": None,
            "current_pi_excess_raw": None,
            "current_pi_premium_raw": None,
            "current_pi_renewal_raw": None,
        }
        # Capture the table row following "Current Professional Indemnity Policy"
        pattern = re.compile(
            r"Current Professional Indemnity Policy\s*"
            r"Insurer\s*(?P<insurer>.+?)\s+"
            r"Broker\s*(?P<broker>.+?)\s+"
            r"Limit of Indemnity\s*(?P<limit>.+?)\s+"
            r"Excess\s*(?P<excess>.+?)\s+"
            r"Premium\s*(?P<premium>.+?)\s+"
            r"Renewal Date\s*(?P<renewal>.+)",
            re.IGNORECASE | re.DOTALL
        )
        m = pattern.search(text)
        if not m:
            return res
        res["current_pi_insurer_raw"] = m.group("insurer").strip()
        res["current_pi_broker_raw"] = m.group("broker").strip()
        res["current_pi_limit_raw"] = m.group("limit").strip()
        res["current_pi_excess_raw"] = m.group("excess").strip()
        res["current_pi_premium_raw"] = m.group("premium").strip()
        res["current_pi_renewal_raw"] = m.group("renewal").strip()
        return res

    def _extract_turnover_latest_year(self, text: str) -> Dict[str, Optional[str]]:
        """
        Template has a multi year turnover table by territory.
        For v1 we attempt to grab the last listed column (assumed latest complete year)
        using a rough pattern; later you can refine using real samples.
        """
        # This is intentionally simple and may need tuning
        block_pattern = re.compile(
            r"Breakdown of turnover/fees.*?UK\s*(?P<uk_row>.+?)"
            r"USA/Canada\s*(?P<usa_row>.+?)"
            r"EU\s*(?P<eu_row>.+?)"
            r"Elsewhere\s*(?P<else_row>.+?)"
            r"Total",
            re.IGNORECASE | re.DOTALL,
        )
        m = block_pattern.search(text)
        if not m:
            return {}
        def last_number(row: str) -> Optional[str]:
            nums = re.findall(r"[£\d,\.kKmM]+", row)
            return nums[-1] if nums else None
        return {
            "turnover_latest_uk_raw": last_number(m.group("uk_row")),
            "turnover_latest_usa_canada_raw": last_number(m.group("usa_row")),
            "turnover_latest_eu_raw": last_number(m.group("eu_row")),
            "turnover_latest_elsewhere_raw": last_number(m.group("else_row")),
        }

    def _extract_activity_split(self, text: str) -> Dict[str, Optional[str]]:
        """
        Parse lines like "Architectural Work - New Build   %"
        """
        patterns = {
            "activity_architectural_new_build_pct_raw":
                r"Architectural Work\s*-\s*New Build\s*(\d+)\s*%",
            "activity_architectural_non_structural_refurb_pct_raw":
                r"Architectural Work\s*–?\s*Non-Structural Refurbishment\s*(\d+)\s*%",
            "activity_building_surveys_non_structural_land_pct_raw":
                r"Building Surveys Non-Structural\s*/\s*Land Surveys\s*(\d+)\s*%",
            "activity_civil_engineering_pct_raw":
                r"Civil Engineering\s*(\d+)\s*%",
            "activity_structural_surveys_valuations_pct_raw":
                r"Structural Surveys\s*/\s*Valuations\s*(\d+)\s*%",
            "activity_project_management_pct_raw":
                r"Project Management\s*(\d+)\s*%",
            "activity_project_coordination_pct_raw":
                r"Project Co-Ordination\s*(\d+)\s*%",
            "activity_interior_design_pct_raw":
                r"Interior Design\s*(\d+)\s*%",
            "activity_quantity_surveying_pct_raw":
                r"Quantity Surveying\s*(\d+)\s*%",
            "activity_other_pct_raw":
                r"Other:\s*(?:Please describe:)?\s*(\d+)\s*%",
        }
        res: Dict[str, Optional[str]] = {}
        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            res[key] = m.group(1) if m else None
        # optional: capture description of "Other" work
        other_desc_match = re.search(
            r"Other:\s*(?:Please describe:)?\s*%.*?Description of other work:\s*(.+?)(?:Total:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if other_desc_match:
            res["activity_other_description_raw"] = other_desc_match.group(1).strip()
        else:
            res["activity_other_description_raw"] = None
        return res

    def _extract_contract_type_split(self, text: str) -> Dict[str, Optional[str]]:
        """
        Parse contract type percentage table.
        """
        patterns = {
            "contract_housing_under_3_floors_pct_raw":
                r"Housing\s*\(Under\s*3\s*Floors\)\s*(\d+)\s*%",
            "contract_housing_over_3_floors_pct_raw":
                r"Housing\s*\(Over\s*3\s*Floors\)\s*(\d+)\s*%",
            "contract_office_retail_pct_raw":
                r"Office\s*/\s*Retail\s*(\d+)\s*%",
            "contract_schools_hospitals_municipal_pct_raw":
                r"Schools\s*/\s*Hospitals\s*/\s*Municipal Buildings\s*(\d+)\s*%",
            "contract_roads_highways_pct_raw":
                r"Roads\s*/\s*Highways\s*/\s*Motorways\s*(\d+)\s*%",
            "contract_power_plants_pct_raw":
                r"Power Plants\s*(\d+)\s*%",
            "contract_cladding_glazing_curtain_walling_pct_raw":
                r"Cladding\s*/\s*Glazing\s*/\s*Curtain Walling\s*(\d+)\s*%",
            "contract_other_pct_raw":
                r"Other:\s*(\d+)\s*%",
        }
        res: Dict[str, Optional[str]] = {}
        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            res[key] = m.group(1) if m else None
        # Other description
        other_desc_match = re.search(
            r"Description of other work:\s*(.+?)(?:Total:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        res["contract_other_description_raw"] = (
            other_desc_match.group(1).strip() if other_desc_match else None
        )
        return res

    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        lines = self._lines(text)

        raw: Dict[str, Any] = {}

        # Firm basics
        raw["firm_name"] = self._extract_after_label(
            lines, self.LABELS["firm_name"], max_lookahead=3
        )
        raw["date_established"] = self._extract_after_label(
            lines, self.LABELS["date_established"], max_lookahead=2
        )
        raw["website"] = self._extract_after_label(
            lines, self.LABELS["website"], max_lookahead=2
        )
        raw["email"] = self._extract_after_label(
            lines, self.LABELS["email"], max_lookahead=2
        )
        raw["telephone"] = self._extract_after_label(
            lines, self.LABELS["telephone"], max_lookahead=2
        )

        # Staff
        raw.update(self._extract_staff_block(lines))

        # Current PI policy
        raw.update(self._extract_current_pi_block(text))

        # Turnover latest year
        raw.update(self._extract_turnover_latest_year(text))

        # Activity and contract splits
        raw.update(self._extract_activity_split(text))
        raw.update(self._extract_contract_type_split(text))

        # Claims and circumstances flags (raw text only for now)
        claims_block = re.search(
            r"Has any claim been made or loss suffered.*?If YES, please provide details below(.*?)(?:Are you aware of any of the following\?|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        raw["claims_block_raw"] = claims_block.group(1).strip() if claims_block else None

        circumstances_block = re.search(
            r"Are you aware of any of the following\?(.*?)(?:Name of Principal Signing this form:|DECLARATION|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        raw["circumstances_block_raw"] = (
            circumstances_block.group(1).strip() if circumstances_block else None
        )

        # Normalised view
        norm: Dict[str, Any] = {}

        norm["firm_name"] = raw.get("firm_name")
        norm["date_established"] = parse_date(raw.get("date_established"))
        norm["website"] = raw.get("website")
        norm["email"] = raw.get("email")
        norm["telephone"] = raw.get("telephone")

        norm["staff_principals"] = parse_int(raw.get("staff_principals_raw"))
        norm["staff_qualified"] = parse_int(raw.get("staff_qualified_raw"))
        norm["staff_unqualified"] = parse_int(raw.get("staff_unqualified_raw"))
        norm["staff_others"] = parse_int(raw.get("staff_others_raw"))

        norm["current_pi_insurer"] = raw.get("current_pi_insurer_raw")
        norm["current_pi_broker"] = raw.get("current_pi_broker_raw")
        norm["current_pi_limit_raw"] = raw.get("current_pi_limit_raw")
        norm["current_pi_limit_amount"] = parse_money(raw.get("current_pi_limit_raw"))
        norm["current_pi_excess_raw"] = raw.get("current_pi_excess_raw")
        norm["current_pi_excess_amount"] = parse_money(raw.get("current_pi_excess_raw"))
        norm["current_pi_premium_raw"] = raw.get("current_pi_premium_raw")
        norm["current_pi_premium_amount"] = parse_money(raw.get("current_pi_premium_raw"))
        norm["current_pi_renewal_date"] = parse_date(raw.get("current_pi_renewal_raw"))

        norm["turnover_latest_uk"] = parse_money(raw.get("turnover_latest_uk_raw"))
        norm["turnover_latest_usa_canada"] = parse_money(
            raw.get("turnover_latest_usa_canada_raw")
        )
        norm["turnover_latest_eu"] = parse_money(raw.get("turnover_latest_eu_raw"))
        norm["turnover_latest_elsewhere"] = parse_money(
            raw.get("turnover_latest_elsewhere_raw")
        )

        # Percentages
        def parse_pct(key: str) -> Optional[int]:
            return parse_int(raw.get(key))

        pct_keys = [
            "activity_architectural_new_build_pct_raw",
            "activity_architectural_non_structural_refurb_pct_raw",
            "activity_building_surveys_non_structural_land_pct_raw",
            "activity_civil_engineering_pct_raw",
            "activity_structural_surveys_valuations_pct_raw",
            "activity_project_management_pct_raw",
            "activity_project_coordination_pct_raw",
            "activity_interior_design_pct_raw",
            "activity_quantity_surveying_pct_raw",
            "activity_other_pct_raw",
            "contract_housing_under_3_floors_pct_raw",
            "contract_housing_over_3_floors_pct_raw",
            "contract_office_retail_pct_raw",
            "contract_schools_hospitals_municipal_pct_raw",
            "contract_roads_highways_pct_raw",
            "contract_power_plants_pct_raw",
            "contract_cladding_glazing_curtain_walling_pct_raw",
            "contract_other_pct_raw",
        ]
        for key in pct_keys:
            norm_key = key.replace("_raw", "")
            norm[norm_key] = parse_pct(key)

        norm["activity_other_description"] = raw.get(
            "activity_other_description_raw"
        )
        norm["contract_other_description"] = raw.get(
            "contract_other_description_raw"
        )

        # Claims blocks are kept as raw text for manual / later NLP processing
        norm["has_claims_disclosed"] = bool(
            raw.get("claims_block_raw") and raw["claims_block_raw"].strip()
        )
        norm["has_circumstances_disclosed"] = bool(
            raw.get("circumstances_block_raw") and raw["circumstances_block_raw"].strip()
        )

        return raw, norm


# =============================
# Engine wiring and FastAPI router
# =============================

router = APIRouter()

# Register available extractors
EXTRACTORS: List[TemplateExtractor] = [
    ArchitectsExtractor(),
    # Later: add AccountantsExtractor(), SurveyorsExtractor(), etc
]


def choose_best_extractor(text: str) -> Optional[TemplateExtractor]:
    best: Optional[TemplateExtractor] = None
    best_score = 0.0
    for extractor in EXTRACTORS:
        score = extractor.match_score(text)
        if score > best_score:
            best_score = score
            best = extractor
    # Threshold so we do not apply a template on random noise
    if best_score < 0.5:
        return None
    return best


@router.post("/extract", response_model=ExtractionResponse)
async def extract_endpoint(payload: ExtractionRequest) -> ExtractionResponse:
    text = payload.ocr_text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="ocr_text is required")

    extractor = choose_best_extractor(text)
    if extractor is None:
        raise HTTPException(
            status_code=422,
            detail="No suitable template extractor found for the provided text",
        )

    raw, norm = extractor.extract(text)

    # For many broker templates insurer will be filled in later by quote
    insurer = norm.get("current_pi_insurer") or None
    insurer_confidence = 0.0

    return ExtractionResponse(
        template_id=extractor.template_id,
        profession=extractor.profession,
        insurer=insurer,
        insurer_confidence=insurer_confidence,
        fields_raw=raw,
        fields_normalized=norm,
    )

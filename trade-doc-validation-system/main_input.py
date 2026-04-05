"""
Quinto — FastAPI Backend
Full pipeline: PDF upload → OCR → extraction → cross-doc field checking → output.html
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile, os, logging, json, uuid
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────────────
from ocr_engine import OCREngine, get_ocr_engine
from extractor import DocumentExtractor, get_extractor, ExtractionResult
from field_checker import analyze_all_cross_doc_fields, MASTER_REGISTRY, print_report

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Quinto Document Validation API",
    description="OCR → extraction → cross-doc inconsistency pipeline for trade documents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Serve the two HTML pages as static files if they live next to this script.
# landingpage.html  →  /
# output.html       →  /output.html
STATIC_DIR = Path(__file__).parent
if (STATIC_DIR / "landingpage.html").exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global singletons ──────────────────────────────────────────────────────────
ocr_engine: OCREngine = None
document_extractor: DocumentExtractor = None

# In-memory store for validation results keyed by shipment_id
validation_store: Dict[str, Any] = {}


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global ocr_engine, document_extractor
    logger.info("Initialising OCR Engine …")
    ocr_engine = get_ocr_engine(dpi=300)
    logger.info("Initialising Document Extractor …")
    document_extractor = get_extractor()
    logger.info("Quinto backend ready.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Quinto backend shutting down.")


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ── Serve landing page ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    """Serve landingpage.html at root."""
    lp = STATIC_DIR / "landingpage.html"
    if lp.exists():
        return HTMLResponse(content=lp.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>landingpage.html not found next to main_input.py</h1>", status_code=404)


# ── Serve output page ──────────────────────────────────────────────────────────
@app.get("/output.html", response_class=HTMLResponse)
async def serve_output():
    """Serve output.html (static template — used before a real report is ready)."""
    op = STATIC_DIR / "output.html"
    if op.exists():
        return HTMLResponse(content=op.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>output.html not found</h1>", status_code=404)


# ==============================================================================
# CORE ENDPOINT: /api/v1/validate-shipment
# Accepts 1-5 PDF files, runs the full pipeline, returns structured report.
# ==============================================================================

@app.post("/api/v1/validate-shipment", tags=["Validation"])
async def validate_shipment(
    files: List[UploadFile] = File(...),
    ocr_language: str = "eng",
    enhance_preprocessing: bool = True
):
    """
    Full Quinto pipeline:
      1. Receive PDF files (up to 5 trade documents)
      2. OCR each PDF  →  raw text per page
      3. Classify + extract structured fields  (extractor.py)
      4. Run cross-document field consistency checks  (field_checker.py)
      5. Return JSON validation report + HTML-ready payload

    The JSON response is everything output.html needs to render the report.
    """
    shipment_id = str(uuid.uuid4())[:8].upper()
    logger.info(f"[{shipment_id}] Received {len(files)} file(s)")

    temp_paths = []

    try:
        # ── Step 1 & 2: Save files → OCR ──────────────────────────────────────
        ocr_results: Dict[str, Dict[int, str]] = {}   # filename → {page: text}

        for upload in files:
            if upload.content_type not in ("application/pdf", "application/octet-stream"):
                raise HTTPException(400, f"'{upload.filename}' is not a PDF.")

            tmp_path = os.path.join(tempfile.gettempdir(),
                                    f"{shipment_id}_{upload.filename}")
            contents = await upload.read()
            if len(contents) > 50 * 1024 * 1024:
                raise HTTPException(400, f"'{upload.filename}' exceeds 50 MB limit.")

            with open(tmp_path, "wb") as f:
                f.write(contents)
            temp_paths.append(tmp_path)

            logger.info(f"[{shipment_id}] OCR → {upload.filename}")
            if enhance_preprocessing:
                ocr_results[upload.filename] = ocr_engine.extract_text_enhanced(
                    tmp_path, lang=ocr_language, preprocess=True)
            else:
                ocr_results[upload.filename] = ocr_engine.extract_text(
                    tmp_path, lang=ocr_language)

        # ── Step 3: Extract structured fields from each document ───────────────
        # We collapse all pages of a document into one text blob per file,
        # then classify + extract once per file.
        extracted_documents = []   # list of dicts ready for field_checker.py

        for filename, page_dict in ocr_results.items():
            full_text = "\n".join(page_dict.values())
            result: ExtractionResult = document_extractor.extract(full_text)

            # Map extractor's DocumentType enum value → human-readable label
            # used as "doc_type" key expected by field_checker's MASTER_REGISTRY
            DOC_TYPE_MAP = {
                "commercial_invoice":   "Commercial Invoice",
                "packing_list":         "Packing List",
                "bill_of_lading":       "Bill of Lading",
                "certificate_of_origin":"Certificate of Origin",
                "customs_declaration":  "Customs Declaration",
                "unknown":              filename,   # fallback: use filename
            }
            doc_type_label = DOC_TYPE_MAP.get(result.document_type, filename)

            extracted_documents.append({
                "doc_type":  doc_type_label,
                "filename":  filename,
                "fields":    result.extracted_fields,
                "confidence": result.confidence,
                "quality_score": result.quality_score,
                "warnings":  result.warnings,
            })

            logger.info(
                f"[{shipment_id}] {filename} → {doc_type_label} "
                f"(confidence={result.confidence:.2%}, fields={len(result.extracted_fields)})"
            )
            # DEBUG: Log what was actually extracted
            for fname, fval in result.extracted_fields.items():
                logger.info(f"  {fname} = {fval}")

        # ── Step 4: Cross-document field consistency check (Gemini) ───────────
        logger.info(f"[{shipment_id}] Running cross-doc field checks via Gemini …")
        
        # DEBUG: Log what's being sent to field_checker
        logger.info(f"[{shipment_id}] Extracted Documents Summary:")
        for i, doc in enumerate(extracted_documents):
            logger.info(f"  Doc {i}: {doc['doc_type']} - {len(doc['fields'])} fields")
            for fname, fval in doc['fields'].items():
                logger.info(f"    {fname}: {fval}")
        
        field_check_results = analyze_all_cross_doc_fields(
            extracted_documents, MASTER_REGISTRY
        )
        
        # DEBUG: Log what came back
        logger.info(f"[{shipment_id}] Field check results: {len(field_check_results)} fields analyzed")
        for field_name, result in field_check_results.items():
            has_issues = result.get("inconsistencies_found", False)
            severity = result.get("overall_severity", "UNKNOWN")
            issues_count = len(result.get("issues", []))
            logger.info(f"  - {field_name}: inconsistencies={has_issues}, severity={severity}, issues={issues_count}")
            if "error" in result:
                logger.warning(f"    ERROR: {result['error']}")

        # ── Step 5: Build unified report ──────────────────────────────────────
        issues_flat = []
        total_issues = 0
        critical_count = 0

        for field_name, check in field_check_results.items():
            if check.get("inconsistencies_found"):
                for issue in check.get("issues", []):
                    total_issues += 1
                    sev = issue.get("severity", "LOW")
                    if sev == "CRITICAL":
                        critical_count += 1
                    issues_flat.append({
                        "field":            field_name,
                        "type":             issue.get("type"),
                        "severity":         sev,
                        "documents":        issue.get("documents_involved", []),
                        "details":          issue.get("details"),
                        "values":           issue.get("values", {}),
                        "recommendation":   issue.get("recommendation"),
                    })

        # Compute a simple risk score (0-100)
        sev_weights = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 2}
        raw_score = sum(sev_weights.get(i["severity"], 0) for i in issues_flat)
        risk_score = min(100, raw_score)

        report = {
            "shipment_id":       f"TRD-{datetime.now().year}-{shipment_id}",
            "processed_at":      datetime.now().isoformat(),
            "documents_received": len(extracted_documents),
            "fields_checked":    len(field_check_results),
            "total_issues":      total_issues,
            "critical_issues":   critical_count,
            "risk_score":        risk_score,
            "extracted_documents": extracted_documents,
            "field_check_results": field_check_results,
            "issues":            issues_flat,
        }

        # Store for later retrieval
        validation_store[shipment_id] = report

        logger.info(
            f"[{shipment_id}] Done. risk_score={risk_score}, issues={total_issues}"
        )
        return JSONResponse(content=report, status_code=200)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[{shipment_id}] Pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Pipeline error: {str(e)}")

    finally:
        # Always clean up temp files
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


# ==============================================================================
# CONVENIENCE: retrieve a stored report
# ==============================================================================

@app.get("/api/v1/report/{shipment_id}", tags=["Results"])
async def get_report(shipment_id: str):
    """Retrieve a previously computed validation report by shipment ID."""
    if shipment_id not in validation_store:
        raise HTTPException(404, f"No report found for shipment ID: {shipment_id}")
    return validation_store[shipment_id]


@app.get("/api/v1/reports", tags=["Results"])
async def list_reports():
    """List all stored report IDs."""
    return {
        "total": len(validation_store),
        "shipment_ids": list(validation_store.keys()),
        "timestamp": datetime.now().isoformat(),
    }


# ==============================================================================
# TEST ENDPOINT: Debug with sample data
# ==============================================================================

@app.get("/api/v1/test-analysis", tags=["Testing"])
async def test_analysis():
    """Run analysis on sample documents to test if the system is working."""
    shipment_id = "TEST-SAMPLE-001"
    logger.info(f"[{shipment_id}] Running test analysis with sample data...")
    
    # Import sample documents
    from field_checker import SAMPLE_DOCUMENTS
    
    # Run cross-document field checking
    field_check_results = analyze_all_cross_doc_fields(
        SAMPLE_DOCUMENTS, MASTER_REGISTRY
    )
    
    logger.info(f"[{shipment_id}] Field check results: {len(field_check_results)} fields analyzed")

    # Build issues flat
    issues_flat = []
    total_issues = 0
    critical_count = 0

    for field_name, check in field_check_results.items():
        if check.get("inconsistencies_found"):
            for issue in check.get("issues", []):
                total_issues += 1
                sev = issue.get("severity", "LOW")
                if sev == "CRITICAL":
                    critical_count += 1
                issues_flat.append({
                    "field":            field_name,
                    "type":             issue.get("type"),
                    "severity":         sev,
                    "documents":        issue.get("documents_involved", []),
                    "details":          issue.get("details"),
                    "values":           issue.get("values", {}),
                    "recommendation":   issue.get("recommendation"),
                })

    # Compute risk score
    sev_weights = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 2}
    raw_score = sum(sev_weights.get(i["severity"], 0) for i in issues_flat)
    risk_score = min(100, raw_score)

    report = {
        "shipment_id":       f"TRD-TEST-{datetime.now().year}-{shipment_id}",
        "processed_at":      datetime.now().isoformat(),
        "documents_received": len(SAMPLE_DOCUMENTS),
        "fields_checked":    len(field_check_results),
        "total_issues":      total_issues,
        "critical_issues":   critical_count,
        "risk_score":        risk_score,
        "extracted_documents": SAMPLE_DOCUMENTS,
        "field_check_results": field_check_results,
        "issues":            issues_flat,
        "note": "This is a test run with sample data to verify the analysis system is working"
    }

    logger.info(f"[{shipment_id}] Test analysis complete. risk_score={risk_score}, issues={total_issues}")
    return JSONResponse(content=report, status_code=200)


# ==============================================================================
# Single-document endpoints (unchanged from original, still useful for testing)
# ==============================================================================

@app.post("/api/v1/process", tags=["Processing"])
async def process_single_document(
    file: UploadFile = File(...),
    ocr_language: str = "eng",
    enhance_preprocessing: bool = True
):
    """Process a single PDF — OCR + extraction only (no cross-doc checking)."""
    request_id = str(uuid.uuid4())
    tmp_path = None
    try:
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(400, "Only PDF files are supported.")

        tmp_path = os.path.join(tempfile.gettempdir(), f"{request_id}_{file.filename}")
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(400, "File exceeds 50 MB limit.")

        raw_text_dict = (
            ocr_engine.extract_text_enhanced(tmp_path, lang=ocr_language, preprocess=True)
            if enhance_preprocessing
            else ocr_engine.extract_text(tmp_path, lang=ocr_language)
        )

        extraction_results = document_extractor.extract_batch(raw_text_dict)

        return JSONResponse(content={
            "request_id":  request_id,
            "filename":    file.filename,
            "total_pages": len(raw_text_dict),
            "results": [
                document_extractor.to_dict(r)
                for r in extraction_results.values()
            ],
            "timestamp": datetime.now().isoformat(),
        }, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] {str(e)}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_input:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

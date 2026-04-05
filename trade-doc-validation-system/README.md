# Quinto Document Hub
### AI-Powered Trade Document Validation

Quinto is a full-stack, AI-driven application designed to automate the classification, extraction, and cross-validation of international trade shipping documents.

## Overview

Quinto leverages:
- Tesseract OCR  
- Regex-based heuristic extraction  
- Google Gemini LLM  

The system detects discrepancies such as:
- Weight mismatches  
- Missing HS codes  
- Conflicting consignee details  

---

## ✨ Features

### 🔍 Intelligent OCR Pipeline
- Converts PDFs into pre-processed images  
- Extracts raw text using Tesseract OCR  

### 📄 Automated Document Classification
Identifies:
- Commercial Invoices  
- Packing Lists  
- Bills of Lading  
- Certificates of Origin  
- Customs Declarations  

Uses weighted keyword scoring.

### 🧾 Structured Field Extraction
Extracts:
- Dates  
- Weights  
- HS Codes  
- Values  
- Incoterms  

Using targeted regex patterns.

### 🤖 Cross-Document AI Validation
- Powered by **Gemini 1.5 Flash**
- Detects:
  - Value mismatches  
  - Semantic inconsistencies  
  - Missing required fields  
- Based on a strict Master Registry of trade rules  

### 💻 Modern Web UI
- Built with Tailwind CSS and Alpine.js  
- Features:
  - Drag-and-drop upload  
  - Interactive discrepancy dashboard  

---

## 🏗️ Architecture & File Structure

| File | Description |
|------|------------|
| `main_input.py` | FastAPI backend orchestrator (handles uploads, pipeline, frontend serving) |
| `ocr_engine.py` | PDF → image → text extraction using `pdf2image`, `Pillow`, `pytesseract` |
| `extractor.py` | Text normalization, classification, and structured JSON extraction |
| `field_checker.py` | Gemini API interface for cross-document validation |
| `landingpage.html` | Upload UI (served at `/`) |
| `output.html` | Results dashboard (served at `/output.html`) |

---

## ⚙️ Prerequisites

### System Dependencies

#### 1. Poppler (required for `pdf2image`)

- **macOS (Homebrew):**
  ```bash
  brew install poppler
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install poppler-utils
  ```
- **Windows:**
  - Download Poppler binaries
  - Add them to your system PATH

#### 2. Tesseract OCR (required for `pytesseract`)

- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **Windows:**
  - Install Tesseract OCR
  - Add it to your system PATH

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
cd /path/to/your/quinto/repo
```

### 2. Install Python Dependencies
```bash
pip install fastapi uvicorn python-multipart pytesseract pdf2image pillow google-generativeai
```

### 3. Configure Gemini API Key
A sample API key has already been hardcoded in `field_checker.py` for quick testing and evaluation. 
You may replace it if needed:
```python
API_KEY = "YOUR_API_KEY"
```

**Recommended (Production):**
- Use environment variables instead of hardcoding

---

## 💻 Running the Application

### 1. Start the FastAPI Server
```bash
python main_input.py
```
**Alternative (Uvicorn):**
```bash
uvicorn main_input:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open the Web Interface
```text
http://localhost:8000
```
Upload sample trade PDFs to test the pipeline.

---

## 📂 Sample Testing Data (For Judges & Evaluators)

Sample PDFs are included in the repository to demonstrate system capabilities across two testing scenarios.

### Scenario 1: Value Inconsistencies
Documents contain all required fields but include intentional conflicts to test discrepancy detection:
- `Certificate_of_origin.pdf`
- `Commercial_Invoice.pdf`
- `Customs_Declaration.pdf`
- `Packing_List.pdf`

### Scenario 2: Missing Fields
Documents are intentionally incomplete to test missing field detection:
- `Bill_of_Lading.pdf`
- `Customs_Declaration_Missing.pdf`
- `Commercial_Invoice_Missing.pdf`
- `Certificate_of_Origin_Missing.pdf`
- `Shipper_Name_Missing.pdf`

---

## 🛠️ Standalone Testing

Run validation logic without starting the web server:

```bash
# Single-field validation
python field_checker.py

# Full cross-document validation
python field_checker.py --all
```

---

## 📌 Notes
- Ensure Poppler and Tesseract are correctly added to PATH
- OCR accuracy depends on document quality
- Gemini API usage may incur costs

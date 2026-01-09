# Document Intelligence System ✅

An end-to-end document processing pipeline that converts PDFs and scanned images into structured JSON data automatically.

## System Overview

**Input:** Raw document (PDF or image)  
**Output:** Structured JSON with extracted fields, validation, and confidence scores

### Complete Pipeline Stages

✅ **1. Document Ingestion** - Load PDFs/images and convert to standardized format  
✅ **2. Preprocessing** - Deskew, denoise, enhance contrast, binarize  
✅ **3. OCR** - Extract text with positions and confidence scores  
✅ **4. Field Extraction** - Identify invoice fields (number, date, amounts, vendor)  
✅ **5. Validation** - Check data quality and business logic  
✅ **6. JSON Export** - Output clean structured data  

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt
```

### 2. Process a Document

**Command Line:**
```bash
python process_document.py sample_documents/invoice_dataset/image/001.png
```

**Jupyter Notebook:**
```bash
jupyter notebook document_processing_pipeline.ipynb
```

### 3. View Results

Structured JSON output is saved to `output/` folder with:
- Extracted fields (invoice number, date, vendor, amounts)
- Validation report (errors, warnings, quality score)
- OCR metadata (confidence, character count)







### Document Ingestion
- Supports PDF and images (PNG, JPG, TIFF, BMP)
- Converts PDFs to images at 300 DPI using PyMuPDF
- Handles multi-page documents
- No system dependencies required

### Preprocessing
- **Deskewing:** Corrects document rotation using Hough transform
- **Denoising:** Removes scanner artifacts with fastNlMeans
- **Contrast Enhancement:** CLAHE for better text visibility
- **Binarization:** Otsu's thresholding for clean black/white

### OCR
- Uses EasyOCR (pure Python, GPU optional)
- Extracts text with bounding boxes
- Provides confidence scores per text block
- Supports multiple languages

### Field Extraction
- Pattern-based extraction for common fields
- Identifies: invoice number, date, vendor, total, subtotal, tax
- Considers spatial positioning (vendor at top, totals at bottom)
- Extracts all numeric amounts for analysis

### Validation
- Checks field formats (dates, amounts, invoice numbers)
- Validates business logic (subtotal + tax ≈ total)
- Confidence threshold filtering
- Quality score (0-100) calculation
- Generates errors, warnings, and info messages

### Output
- Clean JSON format
- Document metadata
- Extracted fields with confidence
- Validation report
- OCR statistics



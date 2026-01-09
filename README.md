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

## Example Output

```json
{
  "document_info": {
    "filename": "001.png",
    "pages": 1,
    "processed_at": "2026-01-07T22:39:29.709522"
  },
  "extracted_fields": [{
    "invoice_number": "802205",
    "date": "05.08.2007",
    "vendor": "Lopez, Miller and Romero",
    "total": {
      "amount": 141.66,
      "confidence": 0.996
    },
    "subtotal": {
      "amount": 141.66,
      "confidence": 0.996
    },
    "tax": {
      "amount": 10.47,
      "confidence": 0.918
    }
  }],
  "validation": {
    "overall_valid": true,
    "quality_score": 100.0
  },
  "ocr_metadata": {
    "total_text_blocks": 59,
    "average_confidence": 0.93
  }
}
```

## Project Structure

```
ComputerVisionProject/
├── src/
│   ├── ingestion/           # Document loading (PDF/image to PIL Image)
│   ├── preprocessing/       # Image cleaning and enhancement
│   ├── ocr/                # Text extraction with EasyOCR
│   ├── extraction/         # Field identification and structuring
│   └── validation/         # Data quality checks
├── sample_documents/       # Your invoice dataset (7000 images + JSON)
├── output/                 # Generated JSON files
├── process_document.py     # Command-line pipeline
├── document_processing_pipeline.ipynb  # Interactive notebook
└── requirements.txt        # Python dependencies
```

## Features

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

## Usage Examples

### Command Line - Single Document

```bash
python process_document.py sample_documents/invoice_dataset/image/001.png
```

### Python API

```python
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.ocr.ocr_engine import OCREngine
from src.extraction.field_extractor import FieldExtractor
from src.validation.data_validator import DataValidator

# Load document
loader = DocumentLoader(dpi=300)
images = loader.load('invoice.pdf')

# Preprocess
preprocessor = ImagePreprocessor(target_dpi=300)
cleaned = preprocessor.preprocess(images[0])

# OCR
ocr = OCREngine(languages=['en'])
text_data = ocr.extract_text(cleaned)

# Extract fields
extractor = FieldExtractor()
fields = extractor.extract_all_fields(text_data)
formatted = extractor.format_output(fields)

# Validate
validator = DataValidator(min_confidence=0.7)
report = validator.validate_all(formatted)
quality = validator.get_quality_score(formatted, report)

print(f"Quality: {quality}/100")
print(formatted)
```

### Jupyter Notebook

Open `document_processing_pipeline.ipynb` for interactive exploration:
- Cell-by-cell execution
- Visual comparisons (before/after preprocessing)
- Bounding box visualizations
- Confidence color-coding
- Statistics and analysis

## Dataset

Using **Ananthu01/7000_invoice_images_with_json** from Hugging Face:
- 7,000 synthetic invoice images
- Corresponding ground truth JSON files
- Located in `sample_documents/invoice_dataset/`

## Technical Details

### Dependencies
- **Pillow:** Image manipulation
- **PyMuPDF:** PDF to image conversion
- **OpenCV:** Advanced image processing
- **EasyOCR:** Deep learning OCR
- **NumPy:** Array operations
- **Matplotlib:** Visualizations (notebook only)

### No System Dependencies Required!
Unlike other OCR solutions, this project doesn't require:
- ❌ Poppler (we use PyMuPDF instead)
- ❌ Tesseract (we use EasyOCR instead)
- ✅ Everything installs via pip!

## Performance

- **Processing time:** ~5-10 seconds per page (CPU)
- **OCR accuracy:** ~93% average confidence
- **Quality scores:** Typically 90-100/100 for clean invoices
- **GPU acceleration:** Optional for EasyOCR (3-5x faster)

## Troubleshooting

**"Module not found" error:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"NumPy version error":**
```bash
pip install "numpy<2.0" --force-reinstall
```

**Low OCR accuracy:**
- Check image quality and resolution
- Ensure document is properly deskewed
- Try adjusting preprocessing parameters

**Missing fields:**
- Field extraction uses patterns; some invoice formats may differ
- Check validation warnings for hints
- Adjust patterns in `field_extractor.py` if needed

## Next Steps / Extensions

Want to improve the system? Consider:
- Add table extraction for line items
- Train custom models for specific invoice formats
- Add support for more languages
- Implement batch processing for folders
- Create REST API endpoint
- Add database storage for processed documents
- Implement machine learning for field extraction

## License

Educational project - free to use and modify.

## Credits

Built as an end-to-end computer vision document intelligence system.

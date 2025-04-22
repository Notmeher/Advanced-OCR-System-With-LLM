# OCR Enhancement System with AI Vision Models

This repository contains three implementations of an advanced OCR (Optical Character Recognition) system that uses multimodal AI models to improve text extraction from images and PDFs:

1. **Flask Web Application**: A full-featured web interface for processing both images and PDFs with LLaMA Vision
2. **SmolDockling App**: A specialized solution for code extraction using the SmolDocling model
3. **Streamlit + Tesseract + LLaMA Vision**: A hybrid approach combining traditional OCR with AI refinement

## Overview

All three implementations share a common goal: to extract text from images or documents with higher accuracy than traditional OCR alone. They leverage vision-language models (VLMs) to refine and correct text extraction by analyzing both the visual content and initial OCR output.

### Key Features

- **Image preprocessing** options (grayscale conversion, contrast enhancement, noise reduction)
- **Multimodal AI refinement** using LLaMA Vision or SmolDockling models
- **PDF support** (in Flask app) with automatic page extraction and processing
- **User-friendly web interfaces** for easy usage
- **Customizable prompts** to guide the AI refinement process

## Installation

### Prerequisites

- Python 3.8+ 
- [Groq API key](https://console.groq.com/keys) (for LLaMA Vision implementations)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for the Tesseract-based implementation)
- PyTorch (for SmolDockling implementation)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ocr-enhancement-system.git
cd ocr-enhancement-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Groq API key (for LLaMA Vision implementations):
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Flask Web Application

The Flask app provides a comprehensive web interface for processing both images and PDFs with LLaMA Vision:

```bash
python app.py
```

Then open your browser to http://127.0.0.1:5000

Features:
- Upload images or PDF files
- Process entire PDFs or select specific pages
- View and download processed text results
- Configure API key via the interface

### SmolDockling Code OCR App

This implementation specializes in extracting code from images using the SmolDockling model:

```bash
streamlit run SmolDockling.py
```

Features:
- Process single or multiple code images
- Downloads and uses the SmolDockling model locally
- GPU acceleration when available
- Markdown output with code formatting preserved
- Batch processing capability for multiple images

### Streamlit + Tesseract + LLaMA Vision App

This hybrid implementation combines traditional Tesseract OCR with LLaMA Vision refinement:

```bash
streamlit run streamlit_tesseract_llama_app.py
```

Features:
- Tesseract OCR configuration options
- Text region detection and visualization
- Compare raw OCR output with AI-refined text
- Download both raw and refined results

## Models Used

The implementations use different vision-language models:

- **Flask app**: `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API
- **SmolDockling app**: `ds4sd/SmolDocling-256M-preview` (runs locally)
- **Tesseract hybrid app**: `llama-3.2-90b-vision-preview` via Groq API

These models are designed to understand both visual content and text, allowing them to correct OCR errors with high accuracy. SmolDockling is specifically optimized for code extraction.

## Customizing Prompts

The LLaMA Vision implementations include customizable prompt templates that control how the AI refines the extracted text. The default prompt instructs the model to:

1. Fix misspelled words
2. Correct misrecognized characters
3. Preserve original formatting
4. Handle unclear text based on context

You can modify these prompts to focus on specific aspects of text correction based on your needs.

## Requirements

```
flask>=2.0.0
streamlit>=1.22.0
opencv-python>=4.5.0
pillow>=9.0.0
numpy>=1.20.0
pytesseract>=0.3.9
groq>=0.4.0
python-dotenv>=0.19.0
PyMuPDF>=1.19.0  # For PDF processing in Flask app
torch>=2.0.0  # For SmolDockling
transformers>=4.30.0  # For SmolDockling
docling-core  # For SmolDockling
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

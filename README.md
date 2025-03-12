# Advanced OCR System with Tesseract & LLaMA Vision

This project combines Tesseract OCR with LLaMA Vision (via Groq API) to create an advanced OCR system that can extract and refine text from images. The system uses Streamlit for a user-friendly interface.

## Features

- **Image Preprocessing**: Apply grayscale conversion, contrast enhancement, and noise reduction
- **Text Region Detection**: Automatically detect text regions in the image
- **OCR Text Extraction**: Extract text using Tesseract OCR
- **Text Refinement**: Refine extracted text using LLaMA Vision's contextual understanding
- **User-friendly Interface**: Easy-to-use web interface with Streamlit

## Demo
![screencapture-localhost-8501-2025-03-12-14_24_27](https://github.com/user-attachments/assets/8b516b1c-3c7b-4f00-beb8-7d003d1dedf2)

![screencapture-localhost-8501-2025-03-12-14_17_25](https://github.com/user-attachments/assets/33624856-8d47-497c-89ed-7f7530c8e8a9)


## How It Works

1. **Upload an image** containing text
2. **Configure preprocessing options** and OCR settings in the sidebar
3. **Process the image** to extract text using Tesseract
4. **Refine the extracted text** using LLaMA Vision
5. **Download the results** as text files

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/advanced-ocr-system.git
   cd advanced-ocr-system
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`

4. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image and adjust settings as needed

4. Click "Process Image" to extract and refine text

## Configuration Options

### Image Preprocessing
- **Grayscale**: Convert image to grayscale
- **Contrast Enhancement**: Improve image contrast
- **Noise Reduction**: Remove noise from the image

### Tesseract OCR
- **Language**: Select OCR language (default: English)
- **Page Segmentation Mode**: Choose how Tesseract segments the page

### LLaMA Vision
- **Prompt Template**: Customize the prompt for text refinement

## Requirements

- Python 3.7+
- Tesseract OCR
- Groq API access (for LLaMA Vision)
- Required Python packages (see `requirements.txt`)

## Files

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `.env`: Environment variables (Groq API key)

## Why Combine Tesseract and LLaMA Vision?

- **Tesseract OCR** is efficient for extracting text from clear images
- **LLaMA Vision** provides contextual understanding to correct errors and handle ambiguous text
- The combination offers better accuracy and robustness compared to either alone

## Limitations

- Performance depends on image quality and clarity
- Requires internet access for LLaMA Vision API
- Processing large images may take time

## Future Improvements

- Add batch processing for multiple images
- Implement more advanced preprocessing techniques
- Add support for additional languages
- Improve text region detection accuracy

## License

MIT License

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Groq API](https://groq.com/)
- [LLaMA Vision Model](https://llama.meta.ai/)
- [Streamlit](https://streamlit.io/)

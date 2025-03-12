import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import io
import base64
from groq import Groq
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Advanced OCR System", layout="wide")
st.title("Advanced OCR System with Tesseract & LLaMA Vision")

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Helper functions
def preprocess_image(image, preprocessing_options):
    """Apply preprocessing techniques to improve image quality"""
    img_array = np.array(image)
    
    # Apply preprocessing based on selected options
    if "grayscale" in preprocessing_options:
        # Check if the image is already grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Convert back to RGB for display purposes
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    if "contrast_enhancement" in preprocessing_options:
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Color image
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            img_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale image
            if len(img_array.shape) == 2:  # Already grayscale
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
                # Convert back to RGB for display purposes
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    if "noise_reduction" in preprocessing_options:
        # Check if the image is already grayscale and convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    
    return Image.fromarray(img_array)

def encode_image_to_base64(image):
    """Convert PIL Image to base64 encoding for API"""
    # Convert image to RGB mode to ensure JPEG compatibility
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def detect_text_regions(image):
    """Use Tesseract to detect text regions"""
    img_array = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        if len(gray.shape) == 3:  # Handle single channel but 3D array
            gray = gray[:,:,0]
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding rectangles for each contour
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small boxes that might be noise
        if w > 15 and h > 15:
            boxes.append((x, y, x + w, y + h))
    
    # If no significant text regions are found, use the entire image
    if not boxes:
        boxes.append((0, 0, img_array.shape[1], img_array.shape[0]))
    
    return boxes, img_array

def extract_text_with_tesseract(image, boxes, lang, config):
    """Extract text from detected regions using Tesseract"""
    img_array = np.array(image)
    extracted_text = ""
    
    # Extract text from each box
    for box in boxes:
        x1, y1, x2, y2 = box
        roi = img_array[y1:y2, x1:x2]
        
        # Skip empty regions
        if roi.size == 0:
            continue
        
        # Convert ROI to PIL Image for Tesseract
        roi_image = Image.fromarray(roi)
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(roi_image, lang=lang, config=config)
        
        if text.strip():
            extracted_text += text + "\n"
    
    return extracted_text

def refine_text_with_llama(text, image, prompt_template):
    """Refine and contextualize extracted text using LLaMA Vision"""
    try:
        # Ensure image is in RGB mode for JPEG encoding
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        img_base64 = encode_image_to_base64(image)
        
        # Prepare the prompt with the extracted text
        prompt = prompt_template.format(text=text)
        
        # Make the API call to LLaMA Vision
        chat_completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error with LLaMA Vision API: {str(e)}")
        st.error("Try using an image with RGB format (JPG) rather than PNG with transparency")
        return text  # Return original text if API fails

def visualize_text_regions(image, boxes):
    """Visualize detected text regions on the image"""
    img_array = np.array(image).copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

# Sidebar for configuration options
st.sidebar.header("Configuration")

# Image preprocessing options
st.sidebar.subheader("Image Preprocessing")
preprocessing_options = st.sidebar.multiselect(
    "Select preprocessing techniques:",
    ["grayscale", "contrast_enhancement", "noise_reduction"],
    default=["grayscale", "contrast_enhancement"]
)

# Tesseract configuration
st.sidebar.subheader("Tesseract OCR Configuration")
ocr_lang = st.sidebar.text_input("Language", "eng")
psm_option = st.sidebar.selectbox(
    "Page Segmentation Mode (PSM)",
    [
        "0 - Orientation and script detection only",
        "1 - Automatic page segmentation with OSD",
        "3 - Fully automatic page segmentation, but no OSD (Default)",
        "4 - Assume a single column of text of variable sizes",
        "6 - Assume a single uniform block of text",
        "7 - Treat the image as a single text line",
        "8 - Treat the image as a single word",
        "9 - Treat the image as a single word in a circle",
        "10 - Treat the image as a single character",
        "11 - Sparse text. Find as much text as possible in no particular order",
        "12 - Sparse text with OSD",
        "13 - Raw line. Treat the image as a single text line"
    ],
    index=2
)
psm = int(psm_option.split(" -")[0])
ocr_config = f'--psm {psm}'

# LLaMA prompt configuration
st.sidebar.subheader("LLaMA Vision Configuration")
prompt_template = st.sidebar.text_area(
    "Prompt Template",
    """I need to improve the text extracted using OCR from an image. Here's the raw OCR output:

```
{text}
```

Please analyze the image and correct any OCR errors. Pay special attention to:
1. Fixing misspelled words
2. Correcting misrecognized characters
3. Preserving the original formatting where possible
4. Handling any unclear or ambiguous text based on context

Return only the corrected text without any explanations."""
)

# Main app area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display original image
    image = Image.open(uploaded_file)
    col1.header("Original Image")
    col1.image(image, use_container_width=True)  # Updated parameter
    
    # Process image button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            try:
                # Step 1: Preprocess the image
                processed_image = preprocess_image(image, preprocessing_options)
                
                # Step 2: Detect text regions
                boxes, _ = detect_text_regions(processed_image)
                
                # Display image with detected regions
                regions_image = visualize_text_regions(processed_image, boxes)
                col2.header("Detected Text Regions")
                col2.image(regions_image, use_container_width=True)  # Updated parameter
                
                # Step 3: Extract text with Tesseract
                extracted_text = extract_text_with_tesseract(processed_image, boxes, ocr_lang, ocr_config)
                
                col1, col2 = st.columns(2)
                
                # Display Tesseract results
                col1.header("Tesseract OCR Output")
                col1.text_area("Extracted Text", extracted_text, height=300)
                
                # Step 4: Refine text with LLaMA Vision
                with st.spinner("Refining text with LLaMA Vision..."):
                    refined_text = refine_text_with_llama(extracted_text, image, prompt_template)
                
                # Display LLaMA Vision results
                col2.header("LLaMA Vision Refined Output")
                col2.text_area("Refined Text", refined_text, height=300)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                # Create temporary files for download
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_ocr:
                    tmp_ocr.write(extracted_text.encode())
                    col1.download_button(
                        "Download OCR Output",
                        data=extracted_text,
                        file_name="ocr_output.txt",
                        mime="text/plain"
                    )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_refined:
                    tmp_refined.write(refined_text.encode())
                    col2.download_button(
                        "Download Refined Output",
                        data=refined_text,
                        file_name="refined_output.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Check the console for full error details.")

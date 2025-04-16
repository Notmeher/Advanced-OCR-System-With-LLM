import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import base64
import requests
import json
import tempfile
import re

# Configure page
st.set_page_config(page_title="Vision OCR System", layout="wide")
st.title("Vision OCR System")

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

def clean_extracted_text(text):
    """Remove any explanatory or commentary text from the extracted content"""
    # Remove lines starting with "Note:", "Please note", etc.
    text = re.sub(r'^Note:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Please note.*$', '', text, flags=re.MULTILINE)
    
    # Remove sentences containing explanatory phrases
    explanatory_phrases = [
        "I have extracted", 
        "The text in the image", 
        "As accurately as possible",
        "The image contains", 
        "I've transcribed", 
        "I've extracted",
        "The quality of the image",
        "The text appears to be",
        "From the image provided",
        "The content of the image",
        "Due to image quality",
        "Here is the text",
        "Text extraction complete",
        "Here's the extracted text",
        "The text from the image is",
        "I've maintained"
    ]
    
    # Remove entire paragraphs that contain explanatory phrases
    for phrase in explanatory_phrases:
        text = re.sub(r'(?m)^.*' + re.escape(phrase) + r'.*$\n?', '', text)
    
    # Remove any blank lines that might have been created
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Trim leading and trailing whitespace
    text = text.strip()
    
    return text

def extract_text_with_ollama(image, prompt_template, model="llama3.2-vision"):
    """Extract text with Ollama using a more basic approach compatible with Ollama's API"""
    try:
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            # Ensure image is in RGB mode for JPEG compatibility
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            image.save(temp_file, format="JPEG")
            temp_file_path = temp_file.name
        
        try:
            # Read the image file as binary
            with open(temp_file_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            # Format for Ollama
            # For Ollama's vision models, we need to create a prompt with markdown image embedding
            payload = {
                "model": model,
                "prompt": f"{prompt_template}\n\n![image](data:image/jpeg;base64,{img_base64})",
                "stream": False
            }
            
            # Make API call to Ollama's generate endpoint
            response = requests.post(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Check for successful response
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response
            raw_text = response_data.get("response", "")
            
            # Clean the text
            cleaned_text = clean_extracted_text(raw_text)
            return cleaned_text
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama server. Make sure Ollama is running.")
        return "Error: Could not connect to Ollama server. Is it running?"
    except Exception as e:
        st.error(f"Error with Ollama API: {str(e)}")
        return f"Error extracting text: {str(e)}"

# Sidebar for configuration options
st.sidebar.header("Configuration")

# Ollama model configuration
st.sidebar.subheader("Ollama Configuration")
model_name = st.sidebar.text_input("Model Name", value="llama3.2-vision")

# Image preprocessing options
st.sidebar.subheader("Image Preprocessing")
preprocessing_options = st.sidebar.multiselect(
    "Select preprocessing techniques:",
    ["grayscale", "contrast_enhancement", "noise_reduction"],
    default=["contrast_enhancement"]
)

# Prompt configuration
st.sidebar.subheader("Vision Model Prompt")
prompt_template = st.sidebar.text_area(
    "Prompt Template",
    """Extract ONLY the text from this image. Do not include any explanations, notes, or commentary of any kind.

You must follow these rules strictly:
1. Return ONLY the extracted text content, exactly as it appears in the image
2. Do not add any introduction, prefix, or suffix
3. Do not include phrases like "Here's the text" or "The image contains"
4. Do not comment on the image quality or your extraction process
5. Do not add any notes or explanations about partially visible text
6. Do not include any metadata about the image

I want solely the raw text content, nothing more. Any explanation will be considered an error."""
)

st.sidebar.subheader("Advanced Options")
preserve_formatting = st.sidebar.checkbox("Preserve Original Formatting", value=True)
extract_tables = st.sidebar.checkbox("Extract Tables (if present)", value=True)

if extract_tables:
    prompt_template += "\n\nIf there are tables, render them in markdown table format. Still provide no explanations."

if not preserve_formatting:
    prompt_template += "\n\nReturn plain text without preserving layout, but still provide no explanations."

# Main app area
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display original image
    image = Image.open(uploaded_file)
    col1.header("Original Image")
    col1.image(image, use_container_width=True)
    
    # Process image button
    if st.button("Extract Text"):
        with st.spinner("Processing image..."):
            try:
                # Step 1: Preprocess the image if options selected
                if preprocessing_options:
                    processed_image = preprocess_image(image, preprocessing_options)
                    col2.header("Processed Image")
                    col2.image(processed_image, use_container_width=True)
                else:
                    processed_image = image
                
                # Step 2: Extract text with Ollama
                with st.spinner(f"Extracting text with Ollama ({model_name})..."):
                    extracted_text = extract_text_with_ollama(
                        processed_image, 
                        prompt_template,
                        model=model_name
                    )
                
                # Print to terminal
                print("\n===== EXTRACTED TEXT OUTPUT =====")
                print(extracted_text)
                print("================================\n")
                
                # Display Ollama results
                st.header("Extracted Text")
                st.text_area("", extracted_text, height=400)
                
                # Download button
                st.download_button(
                    "Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                print(f"ERROR: {error_message}")
                st.error(error_message)
                st.error("Check the console for full error details.")